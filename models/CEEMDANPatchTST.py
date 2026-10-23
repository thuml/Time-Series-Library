"""Single fused backbone, retaining the official PatchTST forecast interface."""
import torch
from torch import nn

from models.PatchTST import Model as PatchTST
from layers.CEEMDAN_decomp import CEEMDANConfig
from layers.CEEMDAN_PatchTST_backbone import ProgressiveEncoder, ProgressiveEncoderLayer


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.decomposition:
            raise ValueError('Use --decomposition 0; progressive_decomp controls the fused backbone')
        if configs.individual:
            raise ValueError('Use --individual 0 to share prediction heads across channels')
        if configs.seq_len < configs.patch_len or configs.stride < 1:
            raise ValueError('Require seq_len>=patch_len and stride>=1')
        if configs.e_layers < 1 or configs.d_model < 1 or configs.n_heads < 1:
            raise ValueError('Require positive depth, width and head count')
        norm = getattr(configs, 'encoder_norm', 'BatchNorm')
        self.base = PatchTST(configs, norm=norm).model
        self.seq_len, self.channels = configs.seq_len, configs.enc_in
        encoder = self.base.backbone
        if encoder.patch_num < 3:
            raise ValueError('Progressive decomposition requires at least 3 patches')
        decomp_config = CEEMDANConfig(
            trials=getattr(configs, 'ceemdan_trials', 8),
            epsilon=getattr(configs, 'ceemdan_epsilon', .005),
            max_imf=getattr(configs, 'ceemdan_max_imf', 3),
            max_sift=getattr(configs, 'ceemdan_max_sift', 50),
            seed=getattr(configs, 'ceemdan_seed', 421))
        encoder.encoder = ProgressiveEncoder([
            ProgressiveEncoderLayer(
                encoder.patch_num, configs.d_model, configs.n_heads, configs.d_ff,
                configs.dropout, norm, getattr(configs, 'progressive_decomp', 'ceemdan'),
                decomp_config, getattr(configs, 'progressive_kernel', 5),
                getattr(configs, 'fusion_attention', 'hybrid'), configs.factor)
            for _ in range(configs.e_layers)])
        # Independent per-depth projections accumulate remainders in forecast
        # space, inspired by Autoformer's decoder trend accumulation. All heads
        # share weights across channels, as does the official flatten head.
        self.trend_heads = nn.ModuleList([
            nn.Sequential(nn.Flatten(start_dim=-2),
                          nn.Linear(self.base.head_nf, configs.pred_len, bias=False),
                          nn.Dropout(configs.head_dropout))
            for _ in range(configs.e_layers)])

    def forward(self, x, return_components=False):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f'Expected [batch, {self.seq_len}, {self.channels}]')
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite model input')
        z = self.base.revin_layer(x, 'norm') if self.base.revin else x
        raw = z.transpose(1, 2)
        if self.base.padding_patch == 'end':
            raw = self.base.padding_patch_layer(raw)
        patches = raw.unfold(-1, self.base.patch_len, self.base.stride)
        encoder = self.base.backbone
        embedding = encoder.W_P(patches)
        b, c, n, d = embedding.shape
        u = encoder.dropout(embedding.reshape(b * c, n, d) + encoder.W_pos)
        seasonal, trends = encoder.encoder(u)
        to_head = lambda t: t.reshape(b, c, n, d).permute(0, 1, 3, 2)
        seasonal_prediction = self.base.head(to_head(seasonal)).transpose(1, 2)
        per_layer_trends = [head(to_head(t)).transpose(1, 2)
                            for head, t in zip(self.trend_heads, trends)]
        trend_prediction = torch.stack(per_layer_trends).sum(0)
        prediction = seasonal_prediction + trend_prediction
        if self.base.revin:
            prediction = self.base.revin_layer(prediction, 'denorm')
        if return_components:
            # Components intentionally use normalized coordinates. Denormalizing
            # them independently would add the RevIN mean more than once.
            return prediction, {'seasonal_normalized': seasonal_prediction,
                                'trend_normalized': trend_prediction,
                                'layer_trends_normalized': per_layer_trends}
        return prediction
