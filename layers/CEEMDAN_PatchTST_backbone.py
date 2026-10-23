"""PatchTST tokens + Autoformer progressive decomposition and lag aggregation."""
import math

import torch
from torch import nn

from layers.PatchTST_backbone import TSTEncoderLayer
from layers.CEEMDAN_decomp import make_decomposition


class InstanceAutoCorrelation(nn.Module):
    """Autoformer FFT/top-k circular delay aggregation, per series in both modes.

    Unlike upstream's batch-shared training delays, this preserves channel
    independence after folding batch and channels together. Delays are measured
    in patch strides. Explicit FFT length also supports odd patch counts.
    """
    def __init__(self, d_model, n_heads, factor=1., dropout=0.):
        super().__init__()
        if factor <= 0:
            raise ValueError('autocorrelation factor must be positive')
        self.n_heads = n_heads
        self.factor = factor
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        split = lambda z: z.reshape(b, n, self.n_heads, d // self.n_heads).permute(0, 2, 3, 1)
        # FFTs on CPU/MPS/CUDA consistently use float32 under mixed precision.
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        corr = torch.fft.irfft(torch.fft.rfft(q.float(), n=n, dim=-1) *
                               torch.fft.rfft(k.float(), n=n, dim=-1).conj(), n=n, dim=-1)
        top_k = min(n, max(1, int(self.factor * math.log(n))))
        scores, delays = corr.mean(dim=(1, 2)).topk(top_k, dim=-1)
        weights = self.dropout(scores.softmax(dim=-1)).to(v.dtype)
        positions = torch.arange(n, device=x.device)[None, :]
        result = torch.zeros_like(v)
        for j in range(top_k):
            indices = ((positions + delays[:, j:j + 1]) % n)[:, None, None, :].expand_as(v)
            result = result + v.gather(-1, indices) * weights[:, j, None, None, None]
        return self.out(result.permute(0, 3, 1, 2).reshape(b, n, d))


class ProgressiveEncoderLayer(TSTEncoderLayer):
    """Two structural decompositions per layer; no raw-input bypass."""
    def __init__(self, q_len, d_model, n_heads, d_ff, dropout, norm,
                 decomposition, decomp_config, kernel_size, attention, factor):
        super().__init__(q_len, d_model, n_heads, d_ff=d_ff, dropout=dropout,
                         norm=norm, res_attention=True)
        if attention not in ('self', 'autocorrelation', 'hybrid'):
            raise ValueError(f'Unknown attention: {attention}')
        self.attention_kind = attention
        if attention != 'self':
            self.autocorrelation = InstanceAutoCorrelation(d_model, n_heads, factor, dropout)
        if attention == 'hybrid':
            self.mix_logit = nn.Parameter(torch.zeros(()))
        self.decomp1 = make_decomposition(decomposition, decomp_config, kernel_size)
        self.decomp2 = make_decomposition(decomposition, decomp_config, kernel_size)

    def forward(self, src, prev=None):
        scores = None
        if self.attention_kind != 'autocorrelation':
            update, _, scores = self.self_attn(src, src, src, prev=prev)
        if self.attention_kind == 'autocorrelation':
            update = self.autocorrelation(src)
        elif self.attention_kind == 'hybrid':
            gate = self.mix_logit.sigmoid()
            update = (1 - gate) * update + gate * self.autocorrelation(src)
        # Decompose residuals before normalization, as in Autoformer. Keep the
        # official PatchTST normalization and FFN components on the seasonal path.
        seasonal, trend1 = self.decomp1(src + self.dropout_attn(update))
        seasonal = self.norm_attn(seasonal)
        seasonal, trend2 = self.decomp2(seasonal + self.dropout_ffn(self.ff(seasonal)))
        seasonal = self.norm_ffn(seasonal)
        return seasonal, trend1 + trend2, scores


class ProgressiveEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        scores = None
        trends = []
        for layer in self.layers:
            x, trend, scores = layer(x, prev=scores)
            trends.append(trend)
        return x, trends
