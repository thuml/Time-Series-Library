"""TimeXer, adapted from https://github.com/thuml/TimeXer.

Preserves the official patch/global-token/variate attention architecture.
Adaptations: local attention signatures, config validation and runner interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
from layers.TimeXer_Embed import DataEmbedding_inverted


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.features = configs.features
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, 'use_norm', True)
        self.output_attention = getattr(configs, 'output_attention', False)
        if self.task_name not in {'long_term_forecast', 'short_term_forecast'}:
            raise ValueError('TimeXer supports forecasting tasks only')
        if self.features not in {'M', 'MS', 'S'}:
            raise ValueError('TimeXer features must be M, MS or S')
        if configs.enc_in < 1:
            raise ValueError('enc_in must be positive')
        if self.features == 'S' and configs.enc_in != 1:
            raise ValueError('TimeXer S mode requires enc_in=1')
        if not 0 < configs.patch_len <= self.seq_len or self.pred_len < 1:
            raise ValueError('Require 0 < patch_len <= seq_len and pred_len > 0')
        if configs.d_model < 2 or configs.d_model % 2 or configs.n_heads < 1 or configs.d_model % configs.n_heads:
            raise ValueError('d_model must be even and divisible by n_heads')
        if configs.e_layers < 1:
            raise ValueError('e_layers must be positive')
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        # MS defaults to the official last-channel target. An explicit target
        # list extends it to independent endogenous streams sharing exogenous
        # context; none of the selected targets are included in that context.
        targets = getattr(configs, 'target_indices', None)
        if targets is not None and self.features != 'MS':
            raise ValueError('Explicit target_indices require MS mode')
        targets = [self.enc_in - 1] if targets is None else list(targets)
        if (not targets or any(type(i) is not int or i < 0 or i >= self.enc_in for i in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError('target_indices must be unique valid channel indices')
        exogenous = [i for i in range(self.enc_in) if i not in set(targets)]
        self.register_buffer('target_indices', torch.tensor(targets, dtype=torch.long), persistent=False)
        self.register_buffer('exogenous_indices', torch.tensor(exogenous, dtype=torch.long), persistent=False)
        self.n_vars = len(targets) if configs.features == 'MS' else configs.enc_in
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.index_select(2, self.target_indices).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc.index_select(2, self.exogenous_indices), x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * stdev.index_select(2, self.target_indices)
            dec_out = dec_out + means.index_select(2, self.target_indices)

        return dec_out


    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f'Expected input [batch, {self.seq_len}, {self.enc_in}]')
        if self.features != 'M' and self.exogenous_indices.numel() == 0 and x_mark_enc is None:
            raise ValueError('Single-channel TimeXer requires historical time markers for cross attention')
        if self.features == 'M':
            dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        prediction = dec_out[:, -self.pred_len:, :]
        # The official model disables attention maps. Preserve the runner's tuple contract.
        return (prediction, None) if self.output_attention else prediction
