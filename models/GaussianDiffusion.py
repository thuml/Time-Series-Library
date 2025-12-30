import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, pred_len, seq_len, timesteps=1000, beta_schedule='linear'):
        super().__init__()

        self.denoise_fn = denoise_fn
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.timesteps = timesteps

        # Define beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 2e-2, timesteps)
        elif beta_schedule == 'cosine':
            # Cosine schedule as proposed in DDPM
            s = 0.008
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps) / timesteps
            alpha_cumprod = torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        self.alphas = 1. - self.betas
        self.alpha_cumprods = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprods = torch.sqrt(self.alpha_cumprods)
        self.sqrt_one_minus_alpha_cumprods = torch.sqrt(1. - self.alpha_cumprods)

        # Move to device
        self.to(denoise_fn.args.device)
        # Ensure all beta schedule tensors are on the correct device
        self.betas = self.betas.to(denoise_fn.args.device)
        self.alphas = self.alphas.to(denoise_fn.args.device)
        self.alpha_cumprods = self.alpha_cumprods.to(denoise_fn.args.device)
        self.sqrt_alpha_cumprods = self.sqrt_alpha_cumprods.to(denoise_fn.args.device)
        self.sqrt_one_minus_alpha_cumprods = self.sqrt_one_minus_alpha_cumprods.to(denoise_fn.args.device)

    def add_noise(self, x0, t):
        """Add noise to clean data x0 at time step t"""
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprods[t].to(x0.device)[:, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprods[t].to(x0.device)[:, None, None]

        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    def forward_loss(self, x_hist, y_true):
        """Compute diffusion loss"""
        t = torch.randint(0, self.timesteps, (y_true.shape[0],), device=y_true.device)
        xt, noise = self.add_noise(y_true, t)

        # Our denoise_fn takes (x_noisy, t, x_hist)
        pred_noise = self.denoise_fn(xt, t, x_hist)
        loss = nn.functional.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample_ddpm(self, x_hist, n_samples=1):
        """DDPM sampling"""
        device = x_hist.device
        B, _, C = x_hist.shape

        # Start with random noise
        x = torch.randn(B * n_samples, self.pred_len, C, device=device)

        # Expand history for multiple samples
        x_hist = x_hist.repeat(n_samples, 1, 1)

        for t in reversed(range(0, self.timesteps)):
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = self.denoise_fn(x, t_batch, x_hist)

            # Compute parameters
            alpha_t = self.alphas[t].to(device)
            alpha_cumprod_t = self.alpha_cumprods[t].to(device)
            beta_t = self.betas[t].to(device)

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Update x
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / (torch.sqrt(1 - alpha_cumprod_t))) * pred_noise) + torch.sqrt(beta_t) * noise

        return x

    @torch.no_grad()
    def sample_ddim(self, x_hist, n_samples=1, eta=0.0):
        """DDIM sampling"""
        device = x_hist.device
        B, _, C = x_hist.shape

        # Start with random noise
        x = torch.randn(B * n_samples, self.pred_len, C, device=device)

        # Expand history for multiple samples
        x_hist = x_hist.repeat(n_samples, 1, 1)

        # Create time sequence
        timesteps = list(range(0, self.timesteps, self.timesteps // 20))[::-1]  # 20 steps

        for i, t in enumerate(timesteps):
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = self.denoise_fn(x, t_batch, x_hist)

            # Get alpha values
            alpha_t = self.alpha_cumprods[t].to(device) if t > 0 else torch.tensor(1.0, device=device)

            if i == len(timesteps) - 1:
                # Last step
                alpha_t_prev = torch.tensor(1.0, device=device)
            else:
                t_prev = timesteps[i+1]
                alpha_t_prev = self.alpha_cumprods[t_prev].to(device) if t_prev > 0 else torch.tensor(1.0, device=device)

            # Compute parameters for DDIM
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            sqrt_alpha_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_t_prev)

            # Predict x0
            x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x0 = torch.clamp(x0, -1.0, 1.0)

            # Direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * pred_noise

            # Sample noise
            noise = torch.randn_like(x) if sigma_t > 0 else torch.zeros_like(x)

            # Update x
            x = sqrt_alpha_prev * x0 + dir_xt + sigma_t * noise

        return x
