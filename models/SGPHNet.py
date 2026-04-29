import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGroupEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, mid_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, L, Cg]
        feat = self.net(x.transpose(1, 2))
        avg_pool = feat.mean(dim=-1)
        max_pool = feat.amax(dim=-1)
        return self.out_proj(torch.cat([avg_pool, max_pool], dim=-1))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_class = configs.num_class
        self.num_time_buckets = max(1, int(getattr(configs, 'num_time_buckets', 1)))
        self.state_graph_profile = str(getattr(configs, 'state_graph_profile', 'none')).lower()

        self.group_indices = self._build_group_indices(self.enc_in, self.state_graph_profile)
        self.group_hidden_dim = max(self.d_model // 2, 32)
        self.group_encoders = nn.ModuleList(
            [TemporalGroupEncoder(len(indices), self.group_hidden_dim, self.dropout) for indices in self.group_indices]
        )

        fusion_in_dim = len(self.group_indices) * self.group_hidden_dim + self.enc_in * 2
        fusion_hidden = max(self.d_model, 64)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(fusion_hidden, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.state_head = nn.Linear(self.d_model, self.num_class)
        self.hazard_head = nn.Linear(self.d_model, 1)
        self.time_head = nn.Linear(self.d_model, self.num_time_buckets)
        self.transition_head = nn.Linear(self.d_model, self.num_class * self.num_class)

        graph_mask = self._build_graph_mask(getattr(configs, 'class_names', list(range(self.num_class))))
        self.register_buffer('graph_mask', graph_mask, persistent=False)

    @staticmethod
    def _normalize_raw_label(raw_label):
        if isinstance(raw_label, str):
            raw_label = raw_label.strip()
            if raw_label == '':
                return raw_label
        try:
            numeric = float(raw_label)
            if numeric.is_integer():
                return int(numeric)
            return numeric
        except (TypeError, ValueError):
            return raw_label

    def _build_group_indices(self, enc_in, profile):
        if profile == 'hoister_overspeed' and enc_in == 15:
            return [
                [0, 1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12, 13, 14],
            ]

        num_groups = 4 if enc_in >= 8 else max(1, min(enc_in, 2))
        base = enc_in // num_groups
        remainder = enc_in % num_groups
        groups = []
        start = 0
        for idx in range(num_groups):
            size = base + (1 if idx < remainder else 0)
            groups.append(list(range(start, start + size)))
            start += size
        return [group for group in groups if group]

    def _build_graph_mask(self, class_names):
        mask = torch.ones(self.num_class, self.num_class, dtype=torch.bool)
        if self.state_graph_profile != 'hoister_overspeed':
            return mask

        normalized_names = [self._normalize_raw_label(name) for name in class_names]
        label_to_idx = {label: idx for idx, label in enumerate(normalized_names)}
        allowed_edges = {
            (1, 1), (1, 5), (1, 7),
            (5, 5), (5, 1), (5, 7), (5, 9), (5, 3),
            (7, 7), (7, 5), (7, 1), (7, 9), (7, 3),
            (9, 9), (9, 7), (9, 5), (9, 1), (9, 3),
            (3, 3), (3, 1),
        }

        mask = torch.zeros(self.num_class, self.num_class, dtype=torch.bool)
        for src_raw, dst_raw in allowed_edges:
            if src_raw in label_to_idx and dst_raw in label_to_idx:
                mask[label_to_idx[src_raw], label_to_idx[dst_raw]] = True

        # Always keep self-loops for stability.
        mask |= torch.eye(self.num_class, dtype=torch.bool)
        return mask

    def classification(self, x_enc):
        group_features = []
        for indices, encoder in zip(self.group_indices, self.group_encoders):
            group_x = x_enc[:, :, indices]
            group_features.append(encoder(group_x))

        global_mean = x_enc.mean(dim=1)
        global_last = x_enc[:, -1, :]
        fused = torch.cat(group_features + [global_mean, global_last], dim=-1)
        hidden = self.fusion(fused)

        logits = self.state_head(hidden)
        hazard_logits = self.hazard_head(hidden).squeeze(-1)
        time_logits = self.time_head(hidden)

        transition_logits = self.transition_head(hidden).view(-1, self.num_class, self.num_class)
        valid_scores = transition_logits.masked_fill(~self.graph_mask.unsqueeze(0), -1e9)
        transition_probs = F.softmax(valid_scores, dim=-1)
        current_probs = F.softmax(logits, dim=-1)
        next_state_probs = torch.bmm(current_probs.unsqueeze(1), transition_probs).squeeze(1)
        next_state_log_probs = torch.log(next_state_probs.clamp_min(1e-8))

        raw_transition_probs = F.softmax(transition_logits, dim=-1)
        invalid_transition_penalty = (
            raw_transition_probs * (~self.graph_mask.unsqueeze(0)).float()
        ).sum(dim=-1).mean()

        return {
            'logits': logits,
            'hazard_logits': hazard_logits,
            'time_logits': time_logits,
            'transition_logits': transition_logits,
            'next_state_log_probs': next_state_log_probs,
            'invalid_transition_penalty': invalid_transition_penalty,
        }

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name != 'classification':
            raise NotImplementedError('SGPHNet currently supports classification only.')
        return self.classification(x_enc)
