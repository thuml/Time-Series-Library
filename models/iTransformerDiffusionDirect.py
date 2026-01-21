"""
iTransformer + Diffusion for probabilistic time series forecasting.

Supports multiple parameterizations:
- 'x0': Direct x₀ prediction (original)
- 'epsilon': Noise ε prediction (DDPM standard)
- 'v': Velocity v prediction (recommended, most stable)

v-prediction advantages:
- Balanced signal-to-noise ratio across all timesteps
- No need for clamp() to stabilize predictions
- Better gradient flow during training

Architecture:
    Input x_hist [B, seq_len, N]
        → iTransformer Backbone → z [B, N, d_model] (condition features)
        → Diffusion samples y from noise
        → 1D U-Net predicts x₀/ε/v with FiLM + CrossAttention conditioning
        → y_pred = denormalize(output)

Mathematical definitions for v-prediction:
    v = √ᾱ_t · ε − √(1-ᾱ_t) · x₀
    x₀ = √ᾱ_t · x_t − √(1-ᾱ_t) · v
    ε  = √(1-ᾱ_t) · x_t + √ᾱ_t · v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.Diffusion_layers import UNet1D, ResidualNormalizer


class Model(nn.Module):
    """
    iTransformerDiffusionDirect: 直接预测条件扩散模型

    基于 iTransformer 的条件扩散模型，使用直接预测策略：
    - Stage 1: 训练 iTransformer backbone（MSE 损失）
    - Stage 2: 联合训练确定性预测和扩散去噪网络

    支持的参数化类型：
    - 'x0': 直接预测干净数据（默认，最稳定）
    - 'epsilon': 预测噪声（标准 DDPM）
    - 'v': v-参数化（推荐，更稳定的训练）

    与 iTransformerDiffusion 的区别：
    - 直接预测目标而非噪声，训练更稳定
    - 简化的两阶段训练策略
    - 更好的收敛性质
    - 支持多种参数化类型
    """

    def __init__(self, configs):
        super().__init__()

        # Basic configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model

        # Diffusion configs (with defaults)
        self.timesteps = getattr(configs, "diffusion_steps", 1000)
        self.beta_schedule = getattr(configs, "beta_schedule", "cosine")
        self.cond_dim = getattr(configs, "cond_dim", 256)
        self.unet_channels = getattr(configs, "unet_channels", [64, 128, 256, 512])
        self.n_samples = getattr(configs, "n_samples", 100)

        # Parameterization: 'x0', 'epsilon', or 'v'
        self.parameterization = getattr(configs, "parameterization", "x0")

        # ================== iTransformer Backbone ==================
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # Projection (deterministic prediction for warmup stage)
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # ================== Direct Prediction Diffusion ==================
        # Output normalizer (for stable diffusion training)
        self.output_normalizer = ResidualNormalizer(self.n_vars)

        # 1D U-Net denoiser (predicts x₀ instead of noise)
        self.denoise_net = UNet1D(
            n_vars=self.n_vars,
            pred_len=self.pred_len,
            d_model=self.d_model,
            cond_dim=self.cond_dim,
            channels=self.unet_channels,
        )

        # ================== Diffusion Schedule ==================
        self._setup_diffusion_schedule()

    def _setup_diffusion_schedule(self):
        """Setup beta schedule and precompute diffusion constants."""
        if self.beta_schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, self.timesteps)
        elif self.beta_schedule == "cosine":
            # Cosine schedule (improved DDPM)
            s = 0.008
            steps = self.timesteps + 1
            t = torch.linspace(0, self.timesteps, steps) / self.timesteps
            alpha_cumprod = torch.cos(((t + s) / (1 + s)) * np.pi * 0.5) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

        alphas = 1.0 - betas
        alpha_cumprods = torch.cumprod(alphas, dim=0)

        # Register as buffers (will be moved to correct device automatically)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprods", alpha_cumprods)
        self.register_buffer("sqrt_alpha_cumprods", torch.sqrt(alpha_cumprods))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprods", torch.sqrt(1.0 - alpha_cumprods)
        )

    def backbone_forward(self, x_enc, x_mark_enc=None):
        """
        iTransformer backbone forward pass.

        Args:
            x_enc: [B, seq_len, N] input history
            x_mark_enc: [B, seq_len, M] time marks (optional)
        Returns:
            y_det: [B, pred_len, N] deterministic prediction
            z: [B, N, d_model] encoder features (condition)
            means, stdev: normalization statistics for denormalization
        """
        # Instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        N = self.n_vars

        # Embedding: [B, seq_len, N] -> [B, N, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder: [B, N, d_model] -> [B, N, d_model]
        z, _ = self.encoder(enc_out, attn_mask=None)

        # Handle variate dimension mismatch
        actual_n_vars = z.shape[1]
        if actual_n_vars != N:
            if actual_n_vars > N:
                z = z[:, :N, :]
            else:
                padding = torch.zeros(
                    z.shape[0],
                    N - actual_n_vars,
                    z.shape[2],
                    device=z.device,
                    dtype=z.dtype,
                )
                z = torch.cat([z, padding], dim=1)

        # Projection: [B, N, d_model] -> [B, N, pred_len] -> [B, pred_len, N]
        y_det = self.projection(z).permute(0, 2, 1)

        # Denormalization
        y_det = y_det * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        y_det = y_det + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return y_det, z, means, stdev

    def add_noise(self, x0, t, noise=None):
        """
        Add noise to clean data at time step t.

        Args:
            x0: [B, N, T] clean data
            t: [B] time step indices
            noise: [B, N, T] optional pre-generated noise
        Returns:
            xt: [B, N, T] noisy data
            noise: [B, N, T] the noise added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_cumprod = self.sqrt_alpha_cumprods[t][:, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprods[t][
            :, None, None
        ]

        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    def forward_loss(self, x_enc, x_mark_enc, y_true, stage="joint"):
        """
        Compute training loss.

        Args:
            x_enc: [B, seq_len, N] input history
            x_mark_enc: [B, seq_len, M] time marks
            y_true: [B, pred_len, N] ground truth
            stage: 'warmup' for stage 1 (MSE only), 'joint' for stage 2 (x₀ diffusion)
        Returns:
            loss: scalar loss
            loss_dict: dictionary with individual losses
        """
        B = x_enc.shape[0]
        device = x_enc.device

        # Backbone forward
        y_det, z, means, stdev = self.backbone_forward(x_enc, x_mark_enc)

        # MSE loss for deterministic prediction
        loss_mse = F.mse_loss(y_det, y_true)

        if stage == "warmup":
            # Stage 1: Only MSE loss to train backbone
            return loss_mse, {"loss_mse": loss_mse.item()}

        # Stage 2: Direct x₀ prediction diffusion
        # Normalize target for stable diffusion
        y_norm = (y_true - means[:, 0, :].unsqueeze(1)) / stdev[:, 0, :].unsqueeze(1)

        # Permute for U-Net: [B, pred_len, N] -> [B, N, pred_len]
        y_norm = y_norm.permute(0, 2, 1)

        # Sample random time steps
        t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)

        # Add noise to clean target
        noise = torch.randn_like(y_norm)
        y_noisy, _ = self.add_noise(y_norm, t, noise)

        # Predict based on parameterization
        if self.parameterization == "x0":
            # x₀-prediction: predict clean data directly
            target = y_norm
            pred = self.denoise_net(y_noisy, t, z)
        elif self.parameterization == "epsilon":
            # epsilon-prediction: predict noise
            target = noise
            pred = self.denoise_net(y_noisy, t, z)
        elif self.parameterization == "v":
            # v-parameterization: predict v = √ᾱ·ε - √(1-ᾱ)·x₀
            sqrt_alpha_cumprod = self.sqrt_alpha_cumprods[t][:, None, None]
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprods[t][
                :, None, None
            ]
            target = sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * y_norm
            pred = self.denoise_net(y_noisy, t, z)
        else:
            raise ValueError(f"Unsupported parameterization: {self.parameterization}")

        # Diffusion loss
        loss_diff = F.mse_loss(pred, target)

        # Combined loss (λ = 0.5)
        loss_lambda = 0.5
        loss_total = loss_lambda * loss_mse + (1 - loss_lambda) * loss_diff

        return loss_total, {
            "loss_total": loss_total.item(),
            "loss_mse": loss_mse.item(),
            "loss_diff": loss_diff.item(),
        }

    @torch.no_grad()
    def predict_x0_from_output(self, model_output, x_t, t):
        """
        根据模型输出和参数化类型预测 x₀。

        Args:
            model_output: 模型直接输出
            x_t: [B, N, T] 当前时间步的噪声数据
            t: [B] 时间步
        Returns:
            x0_pred: [B, N, T] 预测的干净数据
        """
        if self.parameterization == "x0":
            # x₀-prediction: 模型直接输出 x₀
            return model_output
        elif self.parameterization == "epsilon":
            # epsilon-prediction: 从噪声预测 x₀
            alpha_cumprod = self.alpha_cumprods[t][:, None, None]
            return (x_t - torch.sqrt(1 - alpha_cumprod) * model_output) / torch.sqrt(
                alpha_cumprod + 1e-8
            )
        elif self.parameterization == "v":
            # v-parameterization: 从 v 预测 x₀
            alpha_cumprod = self.alpha_cumprods[t][:, None, None]
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
            return (
                sqrt_alpha_cumprod * x_t - sqrt_one_minus_alpha_cumprod * model_output
            )
        else:
            raise ValueError(f"不支持的参数化类型: {self.parameterization}")

    @torch.no_grad()
    def sample_ddpm(self, z, n_samples=1):
        """
        DDPM 采样，支持所有参数化类型。

        Args:
            z: [B, N, d_model] 编码器特征（条件）
            n_samples: 每个输入的采样数量
        Returns:
            samples: [n_samples, B, N, pred_len] 采样的预测结果（归一化）
        """
        B, _, _ = z.shape
        device = z.device
        N = self.n_vars

        all_samples = []
        for _ in range(n_samples):
            # 从纯噪声开始
            x = torch.randn(B, N, self.pred_len, device=device)

            # 逆向扩散
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)

                # 模型预测
                model_output = self.denoise_net(x, t_batch, z)

                # 根据参数化类型预测 x₀
                x0_pred = self.predict_x0_from_output(model_output, x, t_batch)
                x0_pred = torch.clamp(x0_pred, -3.0, 3.0)  # 稳定性

                # 推导噪声预测
                alpha_t = self.alpha_cumprods[t]
                noise_pred = (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(
                    1 - alpha_t + 1e-8
                )

                # DDPM 更新
                alpha = self.alphas[t]
                beta = self.betas[t]

                coef1 = 1.0 / torch.sqrt(alpha)
                coef2 = beta / self.sqrt_one_minus_alpha_cumprods[t]
                mean = coef1 * (x - coef2 * noise_pred)

                # Add noise (except at t=0)
                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(beta)
                    x = mean + sigma * noise
                else:
                    x = mean

            all_samples.append(x)

        return torch.stack(all_samples, dim=0)  # [n_samples, B, N, pred_len]

    @torch.no_grad()
    def sample_ddim(self, z, n_samples=1, ddim_steps=50, eta=0.0):
        """
        DDIM 采样，支持所有参数化类型（比 DDPM 更快）。

        Args:
            z: [B, N, d_model] 编码器特征（条件）
            n_samples: 每个输入的采样数量
            ddim_steps: DDIM 步数（默认 50）
            eta: DDIM 随机性（0 = 确定性）
        Returns:
            samples: [n_samples, B, N, pred_len] 采样的预测结果（归一化）
        """
        B, _, _ = z.shape
        device = z.device
        N = self.n_vars

        # Create DDIM time sequence
        step_size = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, step_size))[::-1]

        all_samples = []
        for _ in range(n_samples):
            # Start from pure noise
            x = torch.randn(B, N, self.pred_len, device=device)

            for i, t in enumerate(timesteps):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)

                # Predict x₀ directly
                x0_pred = self.denoise_net(x, t_batch, z)
                x0_pred = torch.clamp(x0_pred, -3.0, 3.0)  # Stability

                # Get alpha values
                alpha_t = self.alpha_cumprods[t]
                if i == len(timesteps) - 1:
                    alpha_t_prev = torch.tensor(1.0, device=device)
                else:
                    t_prev = timesteps[i + 1]
                    alpha_t_prev = self.alpha_cumprods[t_prev]

                # Derive noise from x₀
                noise_pred = (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(
                    1 - alpha_t + 1e-8
                )

                # Compute sigma
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_t_prev)
                    / (1 - alpha_t + 1e-8)
                    * (1 - alpha_t / alpha_t_prev)
                )

                # Direction pointing to xt
                dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * noise_pred

                # Sample noise
                noise = torch.randn_like(x) if sigma_t > 0 else torch.zeros_like(x)

                # Update x
                x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt + sigma_t * noise

            all_samples.append(x)

        return torch.stack(all_samples, dim=0)  # [n_samples, B, N, pred_len]

    @torch.no_grad()
    def sample_ddpm_x0_batch(self, z, n_samples=1):
        """
        Batched DDPM sampling with x₀-parameterization.
        Process all samples in parallel for GPU efficiency.

        Args:
            z: [B, N, d_model] encoder features (condition)
            n_samples: number of samples per input
        Returns:
            samples: [n_samples, B, N, pred_len] sampled predictions (normalized)
        """
        B, _, d = z.shape
        device = z.device
        N = self.n_vars

        # Expand z: [B, N, d] -> [n_samples*B, N, d]
        z_expanded = z.unsqueeze(0).expand(n_samples, -1, -1, -1)
        z_expanded = z_expanded.reshape(n_samples * B, N, d)

        # Batch initialization: [n_samples*B, N, pred_len]
        x = torch.randn(n_samples * B, N, self.pred_len, device=device)

        # Single loop for all samples
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((n_samples * B,), t, device=device, dtype=torch.long)

            # Predict x₀ for all samples at once
            x0_pred = self.denoise_net(x, t_batch, z_expanded)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            # Derive noise from x₀
            alpha_t = self.alpha_cumprods[t]
            noise_pred = (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(
                1 - alpha_t + 1e-8
            )

            # DDPM update
            alpha = self.alphas[t]
            beta = self.betas[t]

            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / self.sqrt_one_minus_alpha_cumprods[t]
            mean = coef1 * (x - coef2 * noise_pred)

            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

        # Reshape: [n_samples*B, N, T] -> [n_samples, B, N, T]
        return x.reshape(n_samples, B, N, self.pred_len)

    @torch.no_grad()
    def sample_ddim_x0_batch(self, z, n_samples=1, ddim_steps=50, eta=0.0):
        """
        Batched DDIM sampling with x₀-parameterization.
        Process all samples in parallel for GPU efficiency.

        Args:
            z: [B, N, d_model] encoder features (condition)
            n_samples: number of samples per input
            ddim_steps: number of DDIM steps (default 50)
            eta: DDIM stochasticity (0 = deterministic)
        Returns:
            samples: [n_samples, B, N, pred_len] sampled predictions (normalized)
        """
        B, _, d = z.shape
        device = z.device
        N = self.n_vars

        # Expand z: [B, N, d] -> [n_samples*B, N, d]
        z_expanded = z.unsqueeze(0).expand(n_samples, -1, -1, -1)
        z_expanded = z_expanded.reshape(n_samples * B, N, d)

        # Create DDIM time sequence
        step_size = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, step_size))[::-1]

        # Batch initialization: [n_samples*B, N, pred_len]
        x = torch.randn(n_samples * B, N, self.pred_len, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((n_samples * B,), t, device=device, dtype=torch.long)

            # Predict x₀ for all samples at once
            x0_pred = self.denoise_net(x, t_batch, z_expanded)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)

            # Get alpha values
            alpha_t = self.alpha_cumprods[t]
            if i == len(timesteps) - 1:
                alpha_t_prev = torch.tensor(1.0, device=device)
            else:
                t_prev = timesteps[i + 1]
                alpha_t_prev = self.alpha_cumprods[t_prev]

            # Derive noise from x₀
            noise_pred = (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(
                1 - alpha_t + 1e-8
            )

            # Compute sigma
            sigma_t = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t + 1e-8) * (1 - alpha_t / alpha_t_prev)
            )

            # Direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * noise_pred

            # Sample noise
            noise = torch.randn_like(x) if sigma_t > 0 else torch.zeros_like(x)

            # Update x
            x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt + sigma_t * noise

        # Reshape: [n_samples*B, N, T] -> [n_samples, B, N, T]
        return x.reshape(n_samples, B, N, self.pred_len)

    @torch.no_grad()
    def sample_chunked(
        self, z, n_samples=1, use_ddim=False, ddim_steps=50, eta=0.0, chunk_size=10
    ):
        """
        Chunked sampling to balance speed and memory usage.

        Args:
            z: [B, N, d_model] encoder features (condition)
            n_samples: total number of samples
            use_ddim: use DDIM instead of DDPM
            ddim_steps: number of DDIM steps
            eta: DDIM stochasticity
            chunk_size: samples per chunk (tune based on GPU memory)
        Returns:
            samples: [n_samples, B, N, pred_len] sampled predictions
        """
        all_samples = []

        for i in range(0, n_samples, chunk_size):
            chunk_n = min(chunk_size, n_samples - i)

            if use_ddim:
                samples = self.sample_ddim_x0_batch(z, chunk_n, ddim_steps, eta)
            else:
                samples = self.sample_ddpm_x0_batch(z, chunk_n)

            all_samples.append(samples)

        return torch.cat(all_samples, dim=0)

    @torch.no_grad()
    def predict(
        self,
        x_enc,
        x_mark_enc=None,
        n_samples=None,
        use_ddim=False,
        ddim_steps=50,
        use_batch_sampling=True,
        chunk_size=10,
    ):
        """
        Probabilistic prediction.

        Args:
            x_enc: [B, seq_len, N] input history
            x_mark_enc: [B, seq_len, M] time marks
            n_samples: number of samples (default: self.n_samples)
            use_ddim: use DDIM instead of DDPM
            ddim_steps: number of DDIM steps
            use_batch_sampling: use batched parallel sampling (faster)
            chunk_size: samples per chunk for batch sampling (tune for GPU memory)
        Returns:
            mean_pred: [B, pred_len, N] mean prediction
            std_pred: [B, pred_len, N] prediction uncertainty
            samples: [n_samples, B, pred_len, N] all samples
        """
        if n_samples is None:
            n_samples = self.n_samples

        # Backbone forward (only for condition features)
        _, z, means, stdev = self.backbone_forward(x_enc, x_mark_enc)

        # Sample predictions (normalized)
        if use_batch_sampling:
            pred_samples = self.sample_chunked(
                z, n_samples, use_ddim, ddim_steps, chunk_size=chunk_size
            )
        else:
            if use_ddim:
                pred_samples = self.sample_ddim_x0(z, n_samples, ddim_steps)
            else:
                pred_samples = self.sample_ddpm_x0(z, n_samples)

        # pred_samples: [n_samples, B, N, pred_len] -> [n_samples, B, pred_len, N]
        pred_samples = pred_samples.permute(0, 1, 3, 2)

        # Denormalize: reverse instance normalization
        # stdev: [B, 1, N], means: [B, 1, N]
        pred_samples = pred_samples * stdev[:, 0, :].unsqueeze(0).unsqueeze(2)
        pred_samples = pred_samples + means[:, 0, :].unsqueeze(0).unsqueeze(2)

        # Statistics
        mean_pred = pred_samples.mean(dim=0)
        std_pred = pred_samples.std(dim=0, unbiased=False)

        return mean_pred, std_pred, pred_samples

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Standard forward for compatibility with TSLib training loop.
        Returns deterministic prediction only.
        """
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            y_det, _, _, _ = self.backbone_forward(x_enc, x_mark_enc)
            return y_det[:, -self.pred_len :, :]
        return None

    def freeze_encoder(self):
        """Freeze iTransformer encoder for stage 2 training."""
        for param in self.enc_embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Keep projection trainable
        for param in self.projection.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        """Unfreeze iTransformer encoder."""
        for param in self.enc_embedding.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True

# 为兼容性添加别名
iTransformerDiffusionDirect = Model
