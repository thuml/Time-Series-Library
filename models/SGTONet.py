import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout):
        super().__init__()
        branch_dim = max(hidden_dim // 3, 16)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(branch_dim),
                    nn.GELU(),
                )
                for kernel_size in (3, 5, 9)
            ]
        )
        fused_dim = branch_dim * len(self.branches)
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, L, C]
        x = x.transpose(1, 2)
        branch_feats = [branch(x) for branch in self.branches]
        fused = self.fusion(torch.cat(branch_feats, dim=1))
        seq_hidden = fused.transpose(1, 2)
        avg_pool = seq_hidden.mean(dim=1)
        max_pool = seq_hidden.amax(dim=1)
        global_hidden = self.out_proj(torch.cat([avg_pool, max_pool], dim=-1))
        return seq_hidden, global_hidden


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name != "classification":
            raise NotImplementedError("SGTONet currently supports classification only.")

        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_class = configs.num_class
        self.label_shift = max(0, int(getattr(configs, "label_shift", 0)))
        self.state_graph_profile = str(getattr(configs, "state_graph_profile", "none")).lower()

        self.encoder = MultiScaleTemporalEncoder(self.enc_in, self.d_model, self.dropout)
        self.horizon_embedding = nn.Embedding(max(self.label_shift + 2, 8), self.d_model)
        self.horizon_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.current_head = nn.Linear(self.d_model, self.num_class)
        self.boundary_head = nn.Linear(self.d_model, 1)
        self.transition_head = nn.Linear(self.d_model, self.num_class * self.num_class)
        self.future_gate = nn.Linear(self.d_model, 1)

        self.edge_operators = nn.Parameter(torch.zeros(self.num_class, self.num_class, self.d_model))
        nn.init.xavier_uniform_(self.edge_operators)

        self.future_refiner = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.future_head = nn.Linear(self.d_model, self.num_class)

        graph_mask = self._build_graph_mask(getattr(configs, "class_names", list(range(self.num_class))))
        self.register_buffer("graph_mask", graph_mask, persistent=False)

    @staticmethod
    def _normalize_raw_label(raw_label):
        if isinstance(raw_label, str):
            raw_label = raw_label.strip()
            if raw_label == "":
                return raw_label
        try:
            numeric = float(raw_label)
            if numeric.is_integer():
                return int(numeric)
            return numeric
        except (TypeError, ValueError):
            return raw_label

    def _build_graph_mask(self, class_names):
        mask = torch.ones(self.num_class, self.num_class, dtype=torch.bool)
        if self.state_graph_profile != "hoister_overspeed":
            return mask

        normalized_names = [self._normalize_raw_label(name) for name in class_names]
        label_to_idx = {label: idx for idx, label in enumerate(normalized_names)}
        allowed_edges = {
            (1, 1), (1, 5), (1, 7),
            (5, 1), (5, 5), (5, 7), (5, 9), (5, 3),
            (7, 1), (7, 5), (7, 7), (7, 9), (7, 3),
            (9, 1), (9, 5), (9, 7), (9, 9), (9, 3),
            (3, 1), (3, 3),
        }

        mask = torch.zeros(self.num_class, self.num_class, dtype=torch.bool)
        for src_raw, dst_raw in allowed_edges:
            if src_raw in label_to_idx and dst_raw in label_to_idx:
                mask[label_to_idx[src_raw], label_to_idx[dst_raw]] = True

        mask |= torch.eye(self.num_class, dtype=torch.bool)
        return mask

    def classification(self, x_enc, future_x=None):
        _, hidden = self.encoder(x_enc)
        horizon_idx = torch.full(
            (hidden.size(0),),
            fill_value=min(self.label_shift, self.horizon_embedding.num_embeddings - 1),
            dtype=torch.long,
            device=hidden.device,
        )
        horizon_hidden = self.horizon_embedding(horizon_idx)
        hidden = self.horizon_proj(torch.cat([hidden, horizon_hidden], dim=-1))

        current_logits = self.current_head(hidden)
        current_probs = F.softmax(current_logits, dim=-1)
        boundary_logits = self.boundary_head(hidden).squeeze(-1)
        boundary_gate = torch.sigmoid(boundary_logits).unsqueeze(-1)

        transition_logits = self.transition_head(hidden).view(-1, self.num_class, self.num_class)
        masked_transition_logits = transition_logits.masked_fill(~self.graph_mask.unsqueeze(0), -1e9)
        transition_probs = F.softmax(masked_transition_logits, dim=-1)
        edge_weights = current_probs.unsqueeze(-1) * transition_probs

        operator_delta = torch.einsum("bij,ijd->bd", edge_weights, self.edge_operators)
        future_hidden = hidden + boundary_gate * self.future_refiner(operator_delta)

        transition_prior = torch.bmm(current_probs.unsqueeze(1), transition_probs).squeeze(1)
        prior_gate = torch.sigmoid(self.future_gate(hidden))
        future_logits = self.future_head(future_hidden) + prior_gate * torch.log(transition_prior.clamp_min(1e-8))

        invalid_transition_penalty = (
            F.softmax(transition_logits, dim=-1) * (~self.graph_mask.unsqueeze(0)).float()
        ).sum(dim=-1).mean()

        outputs = {
            "logits": future_logits,
            "future_logits": future_logits,
            "current_logits": current_logits,
            "boundary_logits": boundary_logits,
            "transition_logits": transition_logits,
            "future_hidden": future_hidden,
            "invalid_transition_penalty": invalid_transition_penalty,
        }

        if future_x is not None:
            _, target_future_hidden = self.encoder(future_x)
            outputs["target_future_hidden"] = target_future_hidden

        return outputs

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.classification(x_enc, future_x=x_dec)
