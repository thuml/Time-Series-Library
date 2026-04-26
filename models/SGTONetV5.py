import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SGTONetV3 import Model as SGTONetV3Model


class FlattenPatchTemporalEncoder(nn.Module):
    def __init__(self, in_channels, seq_len, hidden_dim, dropout, patch_len, stride, n_heads, e_layers, d_ff, activation):
        super().__init__()
        self.in_channels = in_channels
        self.patch_len = max(2, int(patch_len))
        self.stride = max(1, int(stride))
        padded_len = seq_len + self.stride
        self.num_patches = max(1, (padded_len - self.patch_len) // self.stride + 1)

        self.patch_proj = nn.Linear(self.patch_len, hidden_dim)
        self.channel_embedding = nn.Parameter(torch.zeros(1, in_channels, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, self.num_patches, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.flatten_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels * self.num_patches * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        nn.init.normal_(self.channel_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def _patchify(self, x):
        means = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - means) / stdev

        x = x.transpose(1, 2)
        x = F.pad(x, (0, self.stride), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.size(2) > self.num_patches:
            patches = patches[:, :, :self.num_patches, :]
        elif patches.size(2) < self.num_patches:
            pad_count = self.num_patches - patches.size(2)
            pad = patches[:, :, -1:, :].expand(-1, -1, pad_count, -1)
            patches = torch.cat([patches, pad], dim=2)
        return patches

    def forward(self, x):
        patches = self._patchify(x)
        tokens = self.patch_proj(patches)
        tokens = tokens + self.channel_embedding + self.position_embedding
        batch_size = tokens.size(0)
        tokens = tokens.reshape(batch_size * self.in_channels, self.num_patches, -1)
        encoded = self.norm(self.encoder(tokens))
        encoded = encoded.reshape(batch_size, self.in_channels, self.num_patches, -1)
        global_hidden = self.flatten_proj(encoded)
        return encoded.reshape(batch_size, self.in_channels * self.num_patches, -1), global_hidden


class Model(SGTONetV3Model):
    def __init__(self, configs):
        super().__init__(configs)
        patch_len = int(getattr(configs, "patch_len", 16))
        patch_stride = int(getattr(configs, "sgto_patch_stride", 0))
        if patch_stride <= 0:
            patch_stride = max(1, patch_len // 2)
        n_heads = int(getattr(configs, "n_heads", 4))
        if self.d_model % n_heads != 0:
            n_heads = 1
        self.encoder = FlattenPatchTemporalEncoder(
            in_channels=self.enc_in,
            seq_len=int(getattr(configs, "seq_len", 96)),
            hidden_dim=self.d_model,
            dropout=self.dropout,
            patch_len=patch_len,
            stride=patch_stride,
            n_heads=n_heads,
            e_layers=int(getattr(configs, "e_layers", 2)),
            d_ff=int(getattr(configs, "d_ff", self.d_model * 2)),
            activation=str(getattr(configs, "activation", "gelu")),
        )
