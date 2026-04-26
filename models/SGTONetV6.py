import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SGTONetV4 import Model as SGTONetV4Model


class Model(SGTONetV4Model):
    """Dual-mode SGTO-Net.

    The future-state classifier stays conservative, while a local patch-attention
    rare trigger is trained separately for evaluation-time rare fault override.
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.dual_rare_fuse_weight = float(getattr(configs, "sgto_dual_rare_fuse_weight", 0.0))
        self.dual_rare_suppress_weight = float(getattr(configs, "sgto_dual_rare_suppress_weight", 0.0))
        self.dual_rare_context = str(getattr(configs, "sgto_dual_rare_context", "attention")).lower()
        if self.dual_rare_context not in {"attention", "mean", "hidden"}:
            raise ValueError("sgto_dual_rare_context must be one of: attention, mean, hidden")

        self.rare_query = nn.Parameter(torch.zeros(self.d_model))
        self.rare_key_proj = nn.Linear(self.d_model, self.d_model)
        self.rare_value_proj = nn.Linear(self.d_model, self.d_model)
        self.rare_context_norm = nn.LayerNorm(self.d_model)

        rare_input_dim = self.d_model * 3 + self.num_class * 2 + 2
        self.local_rare_head = nn.Sequential(
            nn.Linear(rare_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1),
        )
        nn.init.normal_(self.rare_query, mean=0.0, std=0.02)

    def _rare_patch_context(self, patch_tokens):
        keys = self.rare_key_proj(patch_tokens)
        values = self.rare_value_proj(patch_tokens)
        query = self.rare_query.view(1, 1, -1)
        attn_logits = (keys * query).sum(dim=-1) / math.sqrt(float(self.d_model))
        attn = torch.softmax(attn_logits, dim=1)
        context = torch.sum(attn.unsqueeze(-1) * values, dim=1)
        return self.rare_context_norm(context), attn

    def classification(self, x_enc, future_x=None):
        patch_tokens, hidden = self.encoder(x_enc)
        horizon_idx = torch.full(
            (hidden.size(0),),
            fill_value=min(self.label_shift, self.horizon_embedding.num_embeddings - 1),
            dtype=torch.long,
            device=hidden.device,
        )
        hidden = self.horizon_proj(torch.cat([hidden, self.horizon_embedding(horizon_idx)], dim=-1))

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

        expert_delta = torch.einsum("bk,bkd->bd", transition_prior, self.dest_experts(hidden))
        future_hidden = hidden + boundary_gate * self.future_refiner(operator_delta + expert_delta)

        norm_future_hidden = F.normalize(future_hidden, dim=-1)
        norm_prototypes = F.normalize(self.prototypes, dim=-1)
        prototype_logits = self.proto_logit_scale * torch.matmul(norm_future_hidden, norm_prototypes.transpose(0, 1))

        base_future_logits = self.future_head(future_hidden)
        base_future_logits = base_future_logits + torch.sigmoid(self.future_gate(hidden)) * torch.log(
            transition_prior.clamp_min(1e-8)
        )
        base_future_logits = base_future_logits + self.proto_mix_weight * torch.sigmoid(
            self.prototype_gate(hidden)
        ) * prototype_logits
        future_logits = base_future_logits

        rare_binary_logits = None
        rare_attention = None
        if self.rare_class_index >= 0:
            if self.dual_rare_context == "attention":
                rare_context, rare_attention = self._rare_patch_context(patch_tokens)
            elif self.dual_rare_context == "mean":
                rare_context = self.rare_context_norm(patch_tokens.mean(dim=1))
            else:
                rare_context = self.rare_context_norm(hidden)
            rare_similarity = prototype_logits[:, self.rare_class_index:self.rare_class_index + 1]
            rare_features = torch.cat(
                [
                    hidden,
                    future_hidden,
                    rare_context,
                    current_probs,
                    transition_prior,
                    rare_similarity,
                    boundary_logits.unsqueeze(-1),
                ],
                dim=-1,
            )
            rare_binary_logits = self.local_rare_head(rare_features).squeeze(-1)

            if self.dual_rare_fuse_weight > 0.0:
                future_logits = future_logits.clone()
                rare_gate = torch.sigmoid(rare_binary_logits) * boundary_gate.squeeze(-1)
                future_logits[:, self.rare_class_index] = (
                    future_logits[:, self.rare_class_index]
                    + self.dual_rare_fuse_weight * rare_binary_logits * boundary_gate.squeeze(-1)
                )
                if self.dual_rare_suppress_weight > 0.0 and self.num_class > 1:
                    nonrare_mask = torch.ones(self.num_class, device=future_logits.device, dtype=torch.bool)
                    nonrare_mask[self.rare_class_index] = False
                    suppress = self.dual_rare_suppress_weight * rare_gate.unsqueeze(-1)
                    future_logits[:, nonrare_mask] = future_logits[:, nonrare_mask] - suppress / float(self.num_class - 1)

        invalid_transition_penalty = (
            F.softmax(transition_logits, dim=-1) * (~self.graph_mask.unsqueeze(0)).float()
        ).sum(dim=-1).mean()

        outputs = {
            "logits": future_logits,
            "future_logits": future_logits,
            "base_future_logits": base_future_logits,
            "current_logits": current_logits,
            "boundary_logits": boundary_logits,
            "transition_logits": transition_logits,
            "future_hidden": future_hidden,
            "prototype_logits": prototype_logits,
            "prototypes": self.prototypes,
            "rare_gate_logits": rare_binary_logits if rare_binary_logits is not None else torch.zeros(hidden.size(0), device=hidden.device),
            "rare_attention": rare_attention,
            "rare_class_index": self.rare_class_index,
            "invalid_transition_penalty": invalid_transition_penalty,
        }

        if future_x is not None:
            _, target_future_hidden = self.encoder(future_x)
            outputs["target_future_hidden"] = target_future_hidden

        return outputs
