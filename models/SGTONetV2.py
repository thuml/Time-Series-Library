import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SGTONet import MultiScaleTemporalEncoder


class DestinationExpertBank(nn.Module):
    def __init__(self, num_class, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_class, hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(num_class, hidden_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, hidden):
        return torch.einsum("bd,kde->bke", hidden, self.weight) + self.bias.unsqueeze(0)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name != "classification":
            raise NotImplementedError("SGTONetV2 currently supports classification only.")

        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_class = configs.num_class
        self.label_shift = max(0, int(getattr(configs, "label_shift", 0)))
        self.state_graph_profile = str(getattr(configs, "state_graph_profile", "none")).lower()
        self.proto_logit_scale = float(getattr(configs, "sgto_proto_logit_scale", 8.0))
        self.proto_mix_weight = float(getattr(configs, "sgto_proto_mix_weight", 0.35))
        self.rare_boost_scale = float(getattr(configs, "sgto_rare_boost_scale", 1.25))
        self.rare_class_index = self._resolve_rare_class_index(
            getattr(configs, "class_names", list(range(self.num_class))),
            getattr(configs, "minority_raw_label", ""),
        )

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
        self.dest_experts = DestinationExpertBank(self.num_class, self.d_model)

        self.future_refiner = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        self.prototypes = nn.Parameter(torch.randn(self.num_class, self.d_model))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        self.prototype_gate = nn.Linear(self.d_model, 1)
        self.rare_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
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

    def _resolve_rare_class_index(self, class_names, minority_raw_label):
        minority_label = self._normalize_raw_label(minority_raw_label)
        normalized_names = [self._normalize_raw_label(name) for name in class_names]
        if minority_label in normalized_names:
            return normalized_names.index(minority_label)
        return -1

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
        transition_prior = torch.bmm(current_probs.unsqueeze(1), transition_probs).squeeze(1)

        expert_outputs = self.dest_experts(hidden)
        expert_delta = torch.einsum("bk,bkd->bd", transition_prior, expert_outputs)
        future_delta = operator_delta + expert_delta
        future_hidden = hidden + boundary_gate * self.future_refiner(future_delta)

        norm_future_hidden = F.normalize(future_hidden, dim=-1)
        norm_prototypes = F.normalize(self.prototypes, dim=-1)
        prototype_logits = self.proto_logit_scale * torch.matmul(norm_future_hidden, norm_prototypes.transpose(0, 1))

        future_logits = self.future_head(future_hidden)
        future_logits = future_logits + torch.sigmoid(self.future_gate(hidden)) * torch.log(transition_prior.clamp_min(1e-8))
        future_logits = future_logits + self.proto_mix_weight * torch.sigmoid(self.prototype_gate(hidden)) * prototype_logits

        rare_gate_logits = self.rare_gate(torch.cat([hidden, future_hidden], dim=-1)).squeeze(-1)
        if self.rare_class_index >= 0:
            rare_boost = self.rare_boost_scale * torch.sigmoid(rare_gate_logits) * transition_prior[:, self.rare_class_index]
            future_logits[:, self.rare_class_index] = future_logits[:, self.rare_class_index] + rare_boost

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
            "prototype_logits": prototype_logits,
            "prototypes": self.prototypes,
            "rare_gate_logits": rare_gate_logits,
            "rare_class_index": self.rare_class_index,
            "invalid_transition_penalty": invalid_transition_penalty,
        }

        if future_x is not None:
            _, target_future_hidden = self.encoder(future_x)
            outputs["target_future_hidden"] = target_future_hidden

        return outputs

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.classification(x_enc, future_x=x_dec)
