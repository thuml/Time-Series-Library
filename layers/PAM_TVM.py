import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Pyra_hierarchical_mm_tvm import graph_mm as graph_mm_tvm


class PyramidalAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout, normalize_before, q_k_mask, k_q_mask):
        super(PyramidalAttention, self).__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_k * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_fc = nn.Dropout(dropout)
        self.q_k_mask = q_k_mask
        self.k_q_mask = k_q_mask

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = hidden_states
        bsz, seq_len,  _ = hidden_states.size()

        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)

        q = q.view(bsz, seq_len, self.n_head, self.d_k)
        k = k.view(bsz, seq_len, self.n_head, self.d_k)
        q = q.float().contiguous()
        k = k.float().contiguous()
        # attn_weights.size(): (batch_size, L, num_heads, 11)
        attn_weights = graph_mm_tvm(q, k, self.q_k_mask, self.k_q_mask, False, 0)
        attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))

        v = v.view(bsz, seq_len, self.n_head, self.d_k)
        v = v.float().contiguous()
        # is_t1_diagonaled=True
        attn = graph_mm_tvm(attn_weights, v, self.q_k_mask, self.k_q_mask, True, 0)
        attn = attn.reshape(bsz, seq_len, self.n_head * self.d_k).contiguous()
        context = self.dropout_fc(self.fc(attn))
        context += residual

        if not self.normalize_before:
            context = self.layer_norm(context)

        return context

