"""
Diffusion layers for iTransformer + CRD-Net hybrid architecture.

Components:
- SinusoidalPosEmb: Time step embedding
- ConditionProjector: iTransformer features to global condition
- FiLMLayer: Feature-wise Linear Modulation
- VariateCrossAttention: Cross-attention for variate-level conditioning
- ResBlock1D: 1D residual block with dilated convolution
- UNet1D: 1D U-Net denoising network
- ResidualNormalizer: Residual normalization with EMA tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion time steps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: [B] time step indices
        Returns:
            [B, dim] sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConditionProjector(nn.Module):
    """
    Project iTransformer encoder features to global condition vector.
    Fuses time step embedding with encoder features.
    """

    def __init__(self, d_model, cond_dim, time_emb_dim):
        """
        Args:
            d_model: iTransformer encoder output dimension
            cond_dim: condition vector dimension (for FiLM)
            time_emb_dim: time step embedding dimension
        """
        super().__init__()

        # Project encoder features (after variate averaging)
        self.feat_proj = nn.Sequential(
            nn.Linear(d_model, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, z, t_emb):
        """
        Args:
            z: [B, N, d_model] iTransformer encoder output
            t_emb: [B, time_emb_dim] time step embedding
        Returns:
            [B, cond_dim] global condition vector
        """
        # Average across variates
        z_global = z.mean(dim=1)  # [B, d_model]

        # Project and add time embedding
        cond = self.feat_proj(z_global) + self.time_mlp(t_emb)
        return cond


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Applies: gamma * h + beta
    """

    def __init__(self, cond_dim, hidden_dim):
        """
        Args:
            cond_dim: condition vector dimension
            hidden_dim: hidden feature dimension (channels)
        """
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

        # Initialize gamma to 1, beta to 0
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h, cond):
        """
        Args:
            h: [B, C, T] hidden features
            cond: [B, cond_dim] condition vector
        Returns:
            [B, C, T] modulated features
        """
        gamma = self.gamma(cond).unsqueeze(-1)  # [B, C, 1]
        beta = self.beta(cond).unsqueeze(-1)    # [B, C, 1]
        return gamma * h + beta


class VariateCrossAttention(nn.Module):
    """
    Cross-attention layer for variate-level conditioning.
    Allows denoising features to attend to individual variate representations.
    """

    def __init__(self, query_dim, key_dim, n_heads=4, dropout=0.1):
        """
        Args:
            query_dim: dimension of query (denoising features)
            key_dim: dimension of key/value (encoder features, d_model)
            n_heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x, z):
        """
        Args:
            x: [B, C, T] denoising features (query)
            z: [B, N, d_model] encoder features (key/value)
        Returns:
            [B, C, T] attended features
        """
        B, C, T = x.shape

        # Reshape x for attention: [B, T, C]
        x_t = x.permute(0, 2, 1)  # [B, T, C]

        # Compute Q, K, V
        Q = self.q_proj(x_t)  # [B, T, C]
        K = self.k_proj(z)    # [B, N, C]
        V = self.v_proj(z)    # [B, N, C]

        # Multi-head attention
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        K = K.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        V = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, V)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        out = self.out_proj(out)

        # Residual + norm
        out = self.norm(x_t + out)

        # Back to [B, C, T]
        return out.permute(0, 2, 1)


class ResBlock1D(nn.Module):
    """
    1D Residual block with dilated convolution and FiLM conditioning.
    """

    def __init__(self, in_channels, out_channels, cond_dim, dilation=1, groups=8):
        """
        Args:
            in_channels: input channels
            out_channels: output channels
            cond_dim: condition dimension for FiLM
            dilation: dilation rate for convolution
            groups: number of groups for GroupNorm
        """
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)

        self.film = FiLMLayer(cond_dim, out_channels)

        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Args:
            x: [B, C_in, T] input features
            cond: [B, cond_dim] condition vector
        Returns:
            [B, C_out, T] output features
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        h = self.film(h, cond)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class DownBlock1D(nn.Module):
    """Downsampling block: ResBlock + Downsample"""

    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.res = ResBlock1D(in_channels, out_channels, cond_dim)
        self.down = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, cond):
        h = self.res(x, cond)
        return self.down(h), h  # Return downsampled and skip


class UpBlock1D(nn.Module):
    """Upsampling block: Upsample + ResBlock + optional CrossAttention"""

    def __init__(self, in_channels, out_channels, skip_channels, cond_dim, d_model=None, use_cross_attn=False):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        # After concat with skip: in_channels + skip_channels
        self.res = ResBlock1D(in_channels + skip_channels, out_channels, cond_dim)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn and d_model is not None:
            self.cross_attn = VariateCrossAttention(out_channels, d_model)
        else:
            self.cross_attn = None

    def forward(self, x, skip, cond, z=None):
        """
        Args:
            x: [B, C_in, T] input from lower level
            skip: [B, C_skip, T*2] skip connection from encoder
            cond: [B, cond_dim] condition vector
            z: [B, N, d_model] encoder features for cross-attention (optional)
        """
        h = self.up(x)

        # Handle size mismatch due to odd lengths
        if h.shape[-1] != skip.shape[-1]:
            h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))

        h = torch.cat([h, skip], dim=1)
        h = self.res(h, cond)

        if self.use_cross_attn and self.cross_attn is not None and z is not None:
            h = self.cross_attn(h, z)

        return h


class UNet1D(nn.Module):
    """
    1D U-Net for denoising in CRD-Net.

    Architecture:
    - Input: [B, N, pred_len] (N variates as channels)
    - Encoder: 4 DownBlocks with FiLM
    - Bottleneck: ResBlock + CrossAttention
    - Decoder: 4 UpBlocks with FiLM + CrossAttention
    - Output: [B, N, pred_len]
    """

    def __init__(self, n_vars, pred_len, d_model, cond_dim=256,
                 channels=[64, 128, 256, 512], time_emb_dim=128):
        """
        Args:
            n_vars: number of variates (input/output channels)
            pred_len: prediction length (temporal dimension)
            d_model: iTransformer encoder output dimension
            cond_dim: condition dimension
            channels: U-Net channel progression
            time_emb_dim: time embedding dimension
        """
        super().__init__()

        self.n_vars = n_vars
        self.pred_len = pred_len

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        # Condition projector
        self.cond_proj = ConditionProjector(d_model, cond_dim, time_emb_dim)

        # Initial convolution
        self.init_conv = nn.Conv1d(n_vars, channels[0], kernel_size=3, padding=1)

        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.down_blocks.append(DownBlock1D(in_ch, out_ch, cond_dim))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck_res = ResBlock1D(channels[-1], channels[-1], cond_dim)
        self.bottleneck_attn = VariateCrossAttention(channels[-1], d_model)

        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        # Skip channels are from down blocks outputs: channels[1:] reversed
        skip_channels_list = list(reversed(channels[1:]))  # e.g., [512, 256, 128] for channels=[64,128,256,512]
        for i, out_ch in enumerate(reversed_channels[1:]):
            in_ch = reversed_channels[i]
            skip_ch = skip_channels_list[i]
            # Use cross-attention in all up blocks
            self.up_blocks.append(UpBlock1D(in_ch, out_ch, skip_ch, cond_dim, d_model, use_cross_attn=True))

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels[0], n_vars, kernel_size=1)
        )

    def forward(self, x, t, z):
        """
        Args:
            x: [B, N, pred_len] noisy residual (N variates as channels)
            t: [B] time step indices
            z: [B, N, d_model] iTransformer encoder features
        Returns:
            [B, N, pred_len] predicted noise
        """
        # Time embedding
        t_emb = self.time_emb(t)  # [B, time_emb_dim]

        # Global condition (encoder features + time)
        cond = self.cond_proj(z, t_emb)  # [B, cond_dim]

        # Initial conv
        h = self.init_conv(x)  # [B, C0, T]

        # Encoder
        skips = []
        for down in self.down_blocks:
            h, skip = down(h, cond)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck_res(h, cond)
        h = self.bottleneck_attn(h, z)

        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip, cond, z)

        # Final conv
        out = self.final_conv(h)

        return out


class ResidualNormalizer(nn.Module):
    """
    Normalizer for residuals with EMA tracking of statistics.
    Ensures stable diffusion input during training and inference.
    """

    def __init__(self, n_vars, momentum=0.1, eps=1e-5):
        """
        Args:
            n_vars: number of variates
            momentum: EMA momentum for running statistics
            eps: small constant for numerical stability
        """
        super().__init__()
        self.n_vars = n_vars
        self.momentum = momentum
        self.eps = eps

        # Running statistics (per-variate)
        self.register_buffer('running_mean', torch.zeros(1, 1, n_vars))
        self.register_buffer('running_std', torch.ones(1, 1, n_vars))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def normalize(self, residual, update_stats=True):
        """
        Normalize residual.

        Args:
            residual: [B, pred_len, N] residual tensor
            update_stats: whether to update running statistics (training mode)
        Returns:
            [B, pred_len, N] normalized residual
        """
        if update_stats and self.training:
            # Compute batch statistics (per-variate)
            batch_mean = residual.mean(dim=(0, 1), keepdim=True)  # [1, 1, N]
            batch_std = residual.std(dim=(0, 1), keepdim=True) + self.eps  # [1, 1, N]

            # EMA update
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std
                self.num_batches_tracked += 1

            return (residual - batch_mean) / batch_std
        else:
            # Use running statistics for inference
            return (residual - self.running_mean) / self.running_std

    def denormalize(self, residual):
        """
        Denormalize residual back to original scale.

        Args:
            residual: [B, pred_len, N] normalized residual
        Returns:
            [B, pred_len, N] denormalized residual
        """
        return residual * self.running_std + self.running_mean

    def reset_statistics(self):
        """Reset running statistics."""
        self.running_mean.zero_()
        self.running_std.fill_(1.0)
        self.num_batches_tracked.zero_()
