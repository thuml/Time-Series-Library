"""
FIS Interaction layer of FISformer.

B. Haznedar and L. Karacan, "FISformer: Replacing Self-Attention With a Fuzzy
Inference System in Transformer Models for Time Series Forecasting,"
IEEE Transactions on Fuzzy Systems, vol. 34, no. 8, pp. 2437-2450, 2026.
doi: 10.1109/TFUZZ.2026.3690012

The layer is a drop-in replacement for the inner attention module used by
``layers.SelfAttention_Family.AttentionLayer``. Instead of the dot-product
similarity Q K^T, every query/key element pair (q_ij, k_ij) is processed by a
first-order Sugeno fuzzy inference system:

    1) Fuzzification     mu_q(q_ij; r), mu_k(k_ij; r)        (Gaussian MFs, Eq. 5)
    2) Rule generation   pi_ijr = mu_q * mu_k                (product t-norm, Eq. 6)
    3) Normalization     pi~_ijr = pi_ijr / (sum_s pi_ijs + eps)          (Eq. 7)
    4) Defuzzification   c_ijr = w_rq q_ij + w_rk k_ij + b_r             (Eq. 9)
                         f_QK(i, j) = sum_r pi~_ijr c_ijr                (Eq. 10)
    5) FIS interaction   A[:, j] = softmax over tokens of f_QK[:, j]     (Eq. 11)
                         O = A * V   (element-wise)                      (Eq. 12)

Tensor shapes follow ``AttentionLayer``: queries/keys/values are
(B, N, H, E) where B = batch, N = tokens (variates under inverted
tokenization), H = heads, E = d_model / H.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FISInteraction(nn.Module):
    """FIS-based token interaction (replaces scaled dot-product attention).

    Args:
        mask_flag, factor, scale, attention_dropout, output_attention:
            Kept for signature compatibility with the attention modules in
            ``SelfAttention_Family`` so the layer can be swapped in directly.
            ``mask_flag`` / ``factor`` / ``scale`` are not used by the FIS.
            ``attention_dropout`` is not applied (as in the published
            experiments).
        membershipType: membership function type. Only ``'gaussian'`` is
            available in this release.
        input_dim: per-head feature dimension E (= d_model // n_heads).
        num_rules: number of fuzzy rules / MFs per input (R, default 3).
        num_head: number of heads H (must match ``AttentionLayer``).
        record_maps: if True, intermediate FIS tensors of the most recent
            forward pass in eval mode are kept in ``self.last_maps`` (used to
            produce the membership / rule-activation figures of the paper).
    """

    MAP_NAMES = (
        'fuzzified_query', 'fuzzified_key', 'fuzzy_outputs', 'firing_norm',
        'normalized_rule_aggregated', 'summed_rule_aggregated',
        'attention_scores_Local', 'values', 'output',
    )
    PARAM_NAMES = (
        'similarity_weights', 'importance_weights', 'similarity_sigma',
        'importance_sigma', 'rule_weights',
    )

    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, membershipType='gaussian', input_dim=64,
                 num_rules=3, num_head=8, record_maps=False):
        super(FISInteraction, self).__init__()

        if membershipType != 'gaussian':
            raise NotImplementedError(
                "membershipType='{}' is not available in this release; "
                "only 'gaussian' is supported.".format(membershipType))

        self.num_rules = num_rules
        self.input_dim = input_dim
        self.num_head = num_head
        self.membershipType = membershipType
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)  # not applied, see docstring
        self.eps = 1e-6
        self.record_maps = record_maps
        self.last_maps = None

        # Gaussian MF centres (w) and spreads (sigma) for queries and keys.
        # NOTE: parameter names and shapes follow the official implementation
        # (https://github.com/bhaznedar/FISformer) so that checkpoints are interchangeable.
        self.similarity_weights = nn.Parameter(torch.randn(num_head, input_dim, num_rules))  # query MF centres
        self.importance_weights = nn.Parameter(torch.randn(input_dim, num_rules))            # key MF centres (shared over heads)
        self.similarity_sigma = nn.Parameter(torch.ones(num_head, input_dim, num_rules))     # query MF spreads
        self.importance_sigma = nn.Parameter(torch.ones(num_head, input_dim, num_rules))     # key MF spreads

        # First-order Sugeno consequents: [w_q, w_k, b] for every rule.
        self.rule_weights = nn.Parameter(torch.randn(num_head, input_dim, 2 + 1, num_rules))

    @staticmethod
    def fuzzify(x, centers, sigma):
        """Gaussian membership degrees, Eq. (5). x: (..., H, E) -> (..., H, E, R)."""
        x_expanded = x.unsqueeze(-1)
        return torch.exp(-((x_expanded - centers) ** 2) / (2 * sigma ** 2))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 1) Fuzzification
        mu_q = self.fuzzify(queries, self.similarity_weights, self.similarity_sigma)  # (B, N, H, E, R)
        mu_k = self.fuzzify(keys, self.importance_weights, self.importance_sigma)     # (B, N, H, E, R)

        # Augmented input z = [q, k, 1], Eq. (8)
        z = torch.stack((queries, keys, torch.ones_like(keys)), dim=-1).unsqueeze(-2)  # (B, N, H, E, 1, 3)

        # 2) Rule generation (product t-norm), Eq. (6)
        firing = mu_q * mu_k                                                       # (B, N, H, E, R)

        # 3) Normalization, Eq. (7)
        firing_norm = firing / (firing.sum(dim=-1, keepdim=True) + self.eps)

        # 4) Defuzzification: rule consequents and weighted aggregation, Eqs. (9)-(10)
        consequents = torch.matmul(z, self.rule_weights).squeeze(-2)               # (B, N, H, E, R)
        weighted = consequents * firing_norm
        f_qk = weighted.sum(dim=-1)                                                # (B, N, H, E)

        # 5) FIS interaction: softmax along the token axis, element-wise with V, Eqs. (11)-(12)
        A = F.softmax(f_qk, dim=1)
        output = A * values

        if self.record_maps and not self.training:
            self.last_maps = {
                'fuzzified_query': mu_q, 'fuzzified_key': mu_k,
                'fuzzy_outputs': firing, 'firing_norm': firing_norm,
                'normalized_rule_aggregated': weighted,
                'summed_rule_aggregated': f_qk,
                'attention_scores_Local': A, 'values': values, 'output': output,
            }
            self.last_maps = {k: v.detach().cpu() for k, v in self.last_maps.items()}

        if self.output_attention:
            return output.contiguous(), A
        return output.contiguous(), None

    def save_maps(self, folder):
        """Write the recorded FIS tensors and the learned parameters as .npy files."""
        os.makedirs(folder, exist_ok=True)
        if self.last_maps is not None:
            for name, tensor in self.last_maps.items():
                np.save(os.path.join(folder, name + '.npy'), tensor.numpy())
        for name in self.PARAM_NAMES:
            np.save(os.path.join(folder, name + '.npy'), getattr(self, name).detach().cpu().numpy())
