# iTransformerDiffusionDirect 深度重构计划

> **作者**: Claude Code (AI Assistant)
> **日期**: 2025-01-20
> **版本**: v1.0
> **状态**: 规划阶段

---

## 目录

1. [执行摘要](#执行摘要)
2. [当前模型问题诊断](#当前模型问题诊断)
3. [重构方案详解](#重构方案详解)
   - [架构层面优化](#架构层面优化)
   - [扩散过程优化](#扩散过程优化)
   - [训练策略优化](#训练策略优化)
   - [效率优化](#效率优化)
4. [实施路线图](#实施路线图)
5. [优先级排序与建议](#优先级排序与建议)
6. [预期收益分析](#预期收益分析)
7. [风险与缓解措施](#风险与缓解措施)

---

## 执行摘要

### 核心发现

当前 `iTransformerDiffusionDirect` 模型存在以下**本质问题**：

1. **架构割裂**：iTransformer (Attention) 与 UNet1D (CNN) 架构不一致，条件信息传递受限
2. **参数化不稳定**：x₀-prediction 在高噪声时间步表现不稳定，导致预测 std 偏低
3. **采样效率低**：DDPM 需要 1000 步采样，推理时间过长
4. **训练割裂**：两阶段训练导致骨干网络与扩散网络无法联合优化

### 推荐重构路径

```
Phase 1 (基础优化):  v-prediction + 端到端训练 + 时序损失
         ↓
Phase 2 (架构升级):  DiT 替代 UNet + 层级条件注入
         ↓
Phase 3 (前沿技术):  Flow Matching / Consistency Models
```

### 预期收益

| 指标 | 当前值 | 目标值 | 改进幅度 |
|------|--------|--------|----------|
| MSE | 0.5995 | 0.38-0.40 | -35% |
| CRPS | 0.495 | 0.30-0.35 | -35% |
| 采样步数 | 1000 | 50 | -95% |
| 训练时间 | 基准 | -30% | -30% |

---

## 当前模型问题诊断

### 1. 架构层面问题

#### 1.1 骨干-扩散架构断裂 🔴 严重

**问题描述**：
- iTransformer 使用 Self-Attention 处理变量间关系
- UNet1D 使用 CNN 处理时序维度
- 两者架构不一致，梯度流动和特征传递存在瓶颈

**代码位置**：`models/iTransformerDiffusionDirect.py:85-91`

```python
# 当前实现: 两个独立网络
self.encoder = Encoder(...)        # Transformer
self.denoise_net = UNet1D(...)     # CNN

# 条件传递瓶颈
cond = self.cond_proj(z, t_emb)    # z.mean() 丢失变量级信息
```

**影响**：
- 条件信息从 `z: [B, N, d_model]` 压缩为 `cond: [B, cond_dim]`，丢失变量级细节
- CNN 和 Transformer 的感受野和归纳偏置不匹配
- 无法共享权重或复用预训练

#### 1.2 变量通道化设计缺陷 🟠 中等

**问题描述**：
- UNet1D 将 N 个变量作为通道维度处理
- CNN 卷积核在通道间独立作用，缺乏显式的变量间交互

**代码位置**：`layers/Diffusion_layers.py:206-208`

```python
# UNet1D 输入: [B, N, pred_len] (N 作为通道)
self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, ...)
# Conv1d 无法显式建模变量间相关性
```

**影响**：
- 变量 0, 2 的 MSE 显著高于其他变量（实验观察）
- 多变量联合分布建模能力不足

#### 1.3 条件注入方式单一 🟡 轻微

**问题描述**：
- 仅使用 FiLM (全局) + CrossAttn (局部) 两种注入方式
- FiLM 只接收 `z.mean()`，丢失变量级信息

**代码位置**：`layers/Diffusion_layers.py:72-85`

```python
class ConditionProjector:
    def forward(self, z, t_emb):
        z_global = z.mean(dim=1)  # 压缩！[B, N, d] → [B, d]
        cond = self.feat_proj(z_global) + self.time_mlp(t_emb)
        return cond
```

---

### 2. 扩散过程问题

#### 2.1 x₀-prediction 高噪声不稳定 🔴 严重

**问题描述**：
- 当 t → T (高噪声)，x_t ≈ 纯噪声，从中预测 x₀ 是病态问题
- 需要 `clamp(-3, 3)` 强制稳定，说明预测值经常越界

**代码位置**：`models/iTransformerDiffusionDirect.py:270`

```python
x0_pred = self.denoise_net(x, t_batch, z)
x0_pred = torch.clamp(x0_pred, -3.0, 3.0)  # 强制稳定
```

**数学分析**：
```
高噪声时 (t → T):
  x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
  当 ᾱ_t → 0 时，x_t ≈ ε (纯噪声)

  网络需要从 ε 反推 x₀，信噪比极低，误差放大
```

**影响**：
- 预测 std = 0.73 < 真实 std = 1.05（欠拟合）
- 高噪声步的误差累积到低噪声步

#### 2.2 采样效率低 🔴 严重

**问题描述**：
- DDPM 需要 1000 步完整采样
- 每步需要一次完整的 UNet 前向推理
- 批量采样 100 个样本需要 100,000 次前向

**代码位置**：`models/iTransformerDiffusionDirect.py:265-294`

```python
for t in reversed(range(self.timesteps)):  # 1000 次循环
    x0_pred = self.denoise_net(x, t_batch, z)  # 每次完整前向
    ...
```

**影响**：
- 推理时间 ~10-30 秒/batch（取决于 GPU）
- 无法用于实时预测场景

#### 2.3 固定 β schedule 🟡 轻微

**问题描述**：
- 使用预定义的 linear/cosine schedule
- 不能根据数据特性自适应调整

**代码位置**：`models/iTransformerDiffusionDirect.py:96-110`

---

### 3. 训练策略问题

#### 3.1 两阶段训练割裂 🔴 严重

**问题描述**：
- Stage 1: 只训练 backbone (MSE loss)
- Stage 2: 冻结 backbone，只训练 diffusion
- 扩散网络的梯度无法流回 backbone

**代码位置**：`exp/exp_diffusion_forecast.py` (train_stage1, train_stage2)

```python
def train_stage2(self):
    self.model.freeze_encoder()  # 冻结！梯度断开
    for param in self.model.denoise_net.parameters():
        param.requires_grad = True
```

**影响**：
- Backbone 无法学习对扩散有利的特征表示
- 两阶段优化目标不一致

#### 3.2 损失函数过于简单 🟠 中等

**问题描述**：
- 仅使用 MSE 损失
- 未考虑时序结构（趋势、周期性）
- 未考虑变量间相关性

**代码位置**：`models/iTransformerDiffusionDirect.py:232`

```python
loss_diff = F.mse_loss(x0_pred, y_norm)  # 简单 MSE
```

---

## 重构方案详解

### 架构层面优化

#### 方案 A: Diffusion Transformer (DiT) 统一架构 ⭐⭐⭐ 强烈推荐

**核心思想**：用 Transformer 替代 UNet，实现骨干与去噪网络的架构统一

**理论依据**：
- DiT (Peebles & Xie, 2023) 在图像生成中证明 Transformer 可以完全替代 UNet
- Transformer 的自注意力机制天然适合建模变量间关系
- 架构统一便于权重共享和迁移学习

**架构设计**：

```
┌─────────────────────────────────────────────────────────────┐
│                   DiT-iTransformer 架构                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: x_hist [B, seq_len, N]                              │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────────────────────┐                        │
│  │   iTransformer Encoder          │                        │
│  │   (保持不变，提取条件特征 z)      │                        │
│  └─────────────────────────────────┘                        │
│      │                                                      │
│      ▼ z: [B, N, d_model]                                   │
│      │                                                      │
│  ┌─────────────────────────────────┐                        │
│  │   Noisy Target: x_t [B, N, T]   │ ← t ~ U(0, T)          │
│  └─────────────────────────────────┘                        │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                DiT Blocks (×L)                      │    │
│  │  ┌───────────────────────────────────────────────┐  │    │
│  │  │  1. Patchify: [B, N, T] → [B, P, D]           │  │    │
│  │  │  2. AdaLayerNorm(h, c)  ← c = f(z, t_emb)     │  │    │
│  │  │  3. Self-Attention (across patches)            │  │    │
│  │  │  4. Cross-Attention (to z)                     │  │    │
│  │  │  5. FFN + AdaLayerNorm                         │  │    │
│  │  └───────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────────────────────┐                        │
│  │   Unpatchify → Output [B, N, T] │                        │
│  └─────────────────────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键组件实现**：

```python
class AdaptiveLayerNorm(nn.Module):
    """条件自适应层归一化 (AdaLN)"""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2)  # γ 和 β
        )

    def forward(self, x, cond):
        # x: [B, L, D], cond: [B, cond_dim]
        x = self.norm(x)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class DiTBlock(nn.Module):
    """单个 DiT Block"""
    def __init__(self, dim, n_heads, cond_dim, d_model):
        super().__init__()
        self.adaln1 = AdaptiveLayerNorm(dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        self.adaln2 = AdaptiveLayerNorm(dim, cond_dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        self.adaln3 = AdaptiveLayerNorm(dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        # 用于 Cross-Attention 的 K, V 投影
        self.kv_proj = nn.Linear(d_model, dim * 2)

    def forward(self, x, z, cond):
        """
        Args:
            x: [B, P, D] 去噪特征
            z: [B, N, d_model] 编码器特征
            cond: [B, cond_dim] 全局条件 (含时间步)
        """
        # Self-Attention
        h = self.adaln1(x, cond)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-Attention to encoder features
        h = self.adaln2(x, cond)
        k, v = self.kv_proj(z).chunk(2, dim=-1)
        h, _ = self.cross_attn(h, k, v)
        x = x + h

        # FFN
        h = self.adaln3(x, cond)
        h = self.ffn(h)
        x = x + h

        return x


class DiTDenoiser(nn.Module):
    """DiT 去噪网络 (替代 UNet1D)"""
    def __init__(self, n_vars, pred_len, d_model,
                 dim=256, n_layers=6, n_heads=8, patch_size=4):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.n_patches = (n_vars * pred_len) // patch_size

        # Patchify: flatten variates and time, then split into patches
        self.patch_embed = nn.Linear(patch_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim) * 0.02)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        # Condition projection (combine z and t)
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model + dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, n_heads, dim, d_model)
            for _ in range(n_layers)
        ])

        # Final layer
        self.final_norm = nn.LayerNorm(dim)
        self.final_proj = nn.Linear(dim, patch_size)

    def forward(self, x, t, z):
        """
        Args:
            x: [B, N, T] 噪声目标
            t: [B] 时间步
            z: [B, N, d_model] 编码器特征
        Returns:
            [B, N, T] 预测的 x₀ 或 v
        """
        B, N, T = x.shape

        # Flatten and patchify: [B, N, T] → [B, N*T] → [B, P, patch_size]
        x_flat = x.reshape(B, -1)
        x_patches = x_flat.reshape(B, self.n_patches, self.patch_size)

        # Patch embedding + positional embedding
        h = self.patch_embed(x_patches) + self.pos_embed

        # Time embedding
        t_emb = self.time_embed(t)  # [B, dim]

        # Global condition: z.mean() + t_emb
        z_global = z.mean(dim=1)  # [B, d_model]
        cond = self.cond_proj(torch.cat([z_global, t_emb], dim=-1))  # [B, dim]

        # DiT Blocks
        for block in self.blocks:
            h = block(h, z, cond)

        # Final projection
        h = self.final_norm(h)
        out = self.final_proj(h)  # [B, P, patch_size]

        # Reshape back: [B, P, patch_size] → [B, N*T] → [B, N, T]
        out = out.reshape(B, -1).reshape(B, N, T)

        return out
```

**实施步骤**：

1. **创建 `layers/DiT_layers.py`**
   - 实现 `AdaptiveLayerNorm`
   - 实现 `DiTBlock`
   - 实现 `DiTDenoiser`

2. **修改 `models/iTransformerDiffusionDirect.py`**
   - 将 `self.denoise_net = UNet1D(...)` 替换为 `self.denoise_net = DiTDenoiser(...)`
   - 移除 `ResidualNormalizer` (DiT 内部处理归一化)

3. **调整超参数**
   - `dim=256`, `n_layers=6`, `n_heads=8`, `patch_size=4`
   - 可根据显存调整

**预期收益**：
- MSE 改进 15-25%
- 训练稳定性提升
- 便于后续扩展

---

#### 方案 B: 潜在空间扩散 (Latent Diffusion)

**核心思想**：先将时序压缩到低维潜在空间，再做扩散

**理论依据**：
- Stable Diffusion (Rombach et al., 2022) 证明潜在空间扩散效率更高
- 时序数据通常有较强的时间相关性，可以高效压缩

**架构设计**：

```
原始空间扩散 (当前):
  y ∈ R^{B × pred_len × N}  →  直接扩散
  计算复杂度: O(pred_len × N × timesteps)

潜在空间扩散 (改进):
  y → TemporalEncoder → z ∈ R^{B × L × D} → 扩散 → TemporalDecoder → ŷ
  其中 L = pred_len / compression_ratio, D 为潜在维度
  计算复杂度: O(L × D × timesteps), 压缩比 4-8x
```

**关键组件实现**：

```python
class TemporalVAE(nn.Module):
    """时序变分自编码器"""
    def __init__(self, n_vars, seq_len, latent_dim=64, compression=4):
        super().__init__()
        self.compression = compression
        self.latent_len = seq_len // compression

        # Encoder: 下采样 + 变分
        self.encoder = nn.Sequential(
            nn.Conv1d(n_vars, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 128, 4, stride=2, padding=1),  # /2
            nn.SiLU(),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),  # /4
            nn.SiLU(),
        )
        self.fc_mu = nn.Conv1d(256, latent_dim, 1)
        self.fc_var = nn.Conv1d(256, latent_dim, 1)

        # Decoder: 上采样
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 256, 1),
            nn.SiLU(),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),  # x2
            nn.SiLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),   # x4
            nn.SiLU(),
            nn.Conv1d(64, n_vars, 1)
        )

    def encode(self, x):
        """x: [B, N, T] → z: [B, latent_dim, T/compression]"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """z: [B, latent_dim, T/compression] → x: [B, N, T]"""
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


class LatentDiffusionModel(nn.Module):
    """潜在空间扩散模型"""
    def __init__(self, n_vars, pred_len, d_model, latent_dim=64, compression=4):
        super().__init__()

        # 预训练的 VAE (或端到端训练)
        self.vae = TemporalVAE(n_vars, pred_len, latent_dim, compression)

        # iTransformer backbone (不变)
        self.backbone = iTransformerEncoder(...)

        # 潜在空间扩散 (在压缩后的空间)
        latent_len = pred_len // compression
        self.diffusion = DiTDenoiser(
            n_vars=latent_dim,
            pred_len=latent_len,
            d_model=d_model
        )

    def forward_loss(self, x_enc, y_true, stage='joint'):
        # 编码条件
        z = self.backbone(x_enc)

        # 将目标压缩到潜在空间
        y_perm = y_true.permute(0, 2, 1)  # [B, N, T]
        mu, log_var = self.vae.encode(y_perm)
        y_latent = self.vae.reparameterize(mu, log_var)  # [B, D, T/c]

        # 在潜在空间做扩散
        t = torch.randint(0, self.timesteps, (B,), device=device)
        noise = torch.randn_like(y_latent)
        y_noisy = self.add_noise(y_latent, t, noise)

        pred = self.diffusion(y_noisy, t, z)
        loss_diff = F.mse_loss(pred, y_latent)  # 在潜在空间计算损失

        # VAE 重建损失 + KL 散度
        y_recon = self.vae.decode(y_latent)
        loss_recon = F.mse_loss(y_recon, y_perm)
        loss_kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        return loss_diff + 0.1 * loss_recon + 0.001 * loss_kl
```

**实施步骤**：

1. **预训练 VAE** (可选)
   - 在时序数据上训练 TemporalVAE
   - 确保重建质量满足要求

2. **修改扩散目标**
   - 将扩散从原始空间移到潜在空间
   - 调整 DiT/UNet 输入输出维度

3. **端到端微调**
   - VAE + Diffusion 联合训练

**预期收益**：
- 计算量减少 4-8x
- 采样速度提升 4-8x
- 生成质量可能略有下降，需要平衡

---

#### 方案 C: 层级式条件注入

**核心思想**：在去噪网络每一层注入不同粒度的条件信息

**当前问题**：
```python
# 只使用全局条件
cond = ConditionProjector(z.mean())  # z: [B, N, d] → cond: [B, c]
# 丢失了变量级信息！
```

**改进方案**：

```python
class HierarchicalConditioner(nn.Module):
    """层级式条件生成器"""
    def __init__(self, d_model, cond_dim, time_emb_dim, n_vars, pred_len):
        super().__init__()

        # 1. 全局条件 (用于 FiLM)
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # 2. 变量级条件 (用于 Cross-Attention)
        self.variate_proj = nn.Sequential(
            nn.Linear(d_model, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # 3. 时间级条件 (用于 Temporal Attention)
        self.temporal_proj = nn.Sequential(
            nn.Linear(d_model, pred_len),
            nn.SiLU(),
            nn.Linear(pred_len, pred_len * cond_dim)
        )

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.n_vars = n_vars
        self.pred_len = pred_len
        self.cond_dim = cond_dim

    def forward(self, z, t):
        """
        Args:
            z: [B, N, d_model] 编码器特征
            t: [B] 时间步
        Returns:
            global_cond: [B, cond_dim] 全局条件
            variate_cond: [B, N, cond_dim] 变量级条件
            temporal_cond: [B, T, cond_dim] 时间级条件
        """
        B = z.shape[0]
        t_emb = self.time_mlp(t)  # [B, cond_dim]

        # 全局: 变量平均 + 时间步
        global_cond = self.global_proj(z.mean(dim=1)) + t_emb

        # 变量级: 每个变量独立投影
        variate_cond = self.variate_proj(z)  # [B, N, cond_dim]
        # 加入时间步信息
        variate_cond = variate_cond + t_emb.unsqueeze(1)

        # 时间级: 将变量聚合后展开到时间维度
        temporal_cond = self.temporal_proj(z.mean(dim=1))  # [B, T*cond_dim]
        temporal_cond = temporal_cond.view(B, self.pred_len, self.cond_dim)
        temporal_cond = temporal_cond + t_emb.unsqueeze(1)

        return global_cond, variate_cond, temporal_cond


class HierarchicalUNet1D(nn.Module):
    """层级条件注入的 UNet"""
    def __init__(self, ...):
        super().__init__()
        self.conditioner = HierarchicalConditioner(...)

        # 在不同层使用不同粒度的条件
        self.down_blocks = nn.ModuleList([
            DownBlockWithHierarchicalCond(use_global=True, use_variate=False),
            DownBlockWithHierarchicalCond(use_global=True, use_variate=True),
            DownBlockWithHierarchicalCond(use_global=True, use_variate=True),
        ])

        self.bottleneck = BottleneckWithAllCond()  # 使用所有三种条件

        self.up_blocks = nn.ModuleList([
            UpBlockWithHierarchicalCond(use_temporal=True),
            UpBlockWithHierarchicalCond(use_temporal=True),
            UpBlockWithHierarchicalCond(use_temporal=False),
        ])

    def forward(self, x, t, z):
        global_c, variate_c, temporal_c = self.conditioner(z, t)

        # Encoder
        skips = []
        h = self.init_conv(x)
        for down in self.down_blocks:
            h, skip = down(h, global_c, variate_c)
            skips.append(skip)

        # Bottleneck (使用全部条件)
        h = self.bottleneck(h, global_c, variate_c, temporal_c)

        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip, global_c, temporal_c)

        return self.final_conv(h)
```

**实施步骤**：

1. **实现 `HierarchicalConditioner`**
2. **修改 `ResBlock1D` 支持多种条件输入**
3. **修改 `UNet1D` 架构，在不同层使用不同条件**

**预期收益**：
- 变量 0, 2 的 MSE 改善
- 整体 CRPS 改善 10-15%

---

### 扩散过程优化

#### 方案 D: v-prediction 参数化 ⭐⭐⭐ 强烈推荐

**核心思想**：预测 velocity v 而非 x₀ 或 ε，在所有噪声级别都稳定

**数学定义**：
```
给定: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε

定义 velocity:
  v = √ᾱ_t · ε − √(1-ᾱ_t) · x₀

从 v 恢复:
  x₀ = √ᾱ_t · x_t − √(1-ᾱ_t) · v
  ε  = √(1-ᾱ_t) · x_t + √ᾱ_t · v
```

**为什么 v-prediction 更稳定**：

| 时间步 | ε-prediction | x₀-prediction | v-prediction |
|--------|--------------|---------------|--------------|
| t → 0 (低噪声) | 难 (ε 占比小) | 易 (x₀ 主导) | 中等 |
| t → T (高噪声) | 易 (ε 主导) | 难 (x₀ 占比小) | 中等 |
| 信噪比变化 | 剧烈 | 剧烈 | **平缓** |

**实现代码**：

```python
class VPredictionDiffusion(nn.Module):
    """v-prediction 参数化的扩散模型"""

    def __init__(self, ...):
        super().__init__()
        # ... 其他初始化 ...

    def get_v_target(self, x0, noise, t):
        """计算 v 的目标值"""
        sqrt_alpha = self.sqrt_alpha_cumprods[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprods[t][:, None, None]

        # v = √ᾱ · ε − √(1-ᾱ) · x₀
        v_target = sqrt_alpha * noise - sqrt_one_minus_alpha * x0
        return v_target

    def predict_x0_from_v(self, x_t, v_pred, t):
        """从 v 预测恢复 x₀"""
        sqrt_alpha = self.sqrt_alpha_cumprods[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprods[t][:, None, None]

        # x₀ = √ᾱ · x_t − √(1-ᾱ) · v
        x0_pred = sqrt_alpha * x_t - sqrt_one_minus_alpha * v_pred
        return x0_pred

    def predict_noise_from_v(self, x_t, v_pred, t):
        """从 v 预测恢复 ε"""
        sqrt_alpha = self.sqrt_alpha_cumprods[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprods[t][:, None, None]

        # ε = √(1-ᾱ) · x_t + √ᾱ · v
        eps_pred = sqrt_one_minus_alpha * x_t + sqrt_alpha * v_pred
        return eps_pred

    def forward_loss(self, x_enc, y_true, stage='joint'):
        B = x_enc.shape[0]
        device = x_enc.device

        # Backbone
        y_det, z, means, stdev = self.backbone_forward(x_enc)
        loss_mse = F.mse_loss(y_det, y_true)

        if stage == 'warmup':
            return loss_mse, {'loss_mse': loss_mse.item()}

        # 归一化目标
        y_norm = (y_true - means[:, 0, :].unsqueeze(1)) / stdev[:, 0, :].unsqueeze(1)
        y_norm = y_norm.permute(0, 2, 1)  # [B, N, T]

        # 加噪
        t = torch.randint(0, self.timesteps, (B,), device=device)
        noise = torch.randn_like(y_norm)
        y_noisy, _ = self.add_noise(y_norm, t, noise)

        # 计算 v 目标
        v_target = self.get_v_target(y_norm, noise, t)

        # 预测 v
        v_pred = self.denoise_net(y_noisy, t, z)

        # 损失
        loss_diff = F.mse_loss(v_pred, v_target)

        loss_total = 0.5 * loss_mse + 0.5 * loss_diff
        return loss_total, {
            'loss_total': loss_total.item(),
            'loss_mse': loss_mse.item(),
            'loss_diff': loss_diff.item()
        }

    @torch.no_grad()
    def sample_ddpm_v(self, z, n_samples=1):
        """v-prediction DDPM 采样"""
        B = z.shape[0]
        device = z.device
        N = self.n_vars

        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(B, N, self.pred_len, device=device)

            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)

                # 预测 v
                v_pred = self.denoise_net(x, t_batch, z)

                # 从 v 恢复 x₀ 和 ε
                x0_pred = self.predict_x0_from_v(x, v_pred, t_batch)
                eps_pred = self.predict_noise_from_v(x, v_pred, t_batch)

                # 可选: clamp x0 (v-pred 通常不需要)
                # x0_pred = torch.clamp(x0_pred, -3, 3)

                # DDPM 更新
                alpha = self.alphas[t]
                beta = self.betas[t]

                coef1 = 1.0 / torch.sqrt(alpha)
                coef2 = beta / self.sqrt_one_minus_alpha_cumprods[t]
                mean = coef1 * (x - coef2 * eps_pred)

                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(beta)
                    x = mean + sigma * noise
                else:
                    x = mean

            all_samples.append(x)

        return torch.stack(all_samples, dim=0)
```

**实施步骤**：

1. **修改 `forward_loss`**
   - 计算 v_target 而非 x₀ 或 noise
   - 损失函数改为 `MSE(v_pred, v_target)`

2. **修改采样函数**
   - 从 v_pred 恢复 x₀ 和 ε
   - 使用恢复的 ε 进行 DDPM/DDIM 更新

3. **移除 clamp**
   - v-prediction 通常不需要数值裁剪

**预期收益**：
- 高噪声时间步稳定性提升
- 预测 std 从 0.73 提升到接近 1.05
- MSE 改善 10-20%

---

#### 方案 E: Flow Matching ⭐⭐⭐ 前沿技术

**核心思想**：用最优传输替代扩散过程，学习从噪声到数据的直线路径

**与 DDPM 的本质区别**：

```
DDPM (随机微分方程 SDE):
  dx = f(x,t)dt + g(t)dW
  路径: 曲线，需要 1000 步才能准确积分

Flow Matching (常微分方程 ODE):
  dx = v_θ(x,t)dt
  路径: 直线，50 步即可精确积分
```

**数学推导**：

```
目标: 学习从 p₀ (高斯噪声) 到 p₁ (数据分布) 的映射

最优传输路径 (直线):
  x_t = (1-t) · x₀ + t · x₁
  其中 x₀ ~ N(0, I), x₁ ~ p_data

velocity (路径导数):
  v*(x_t, t) = dx_t/dt = x₁ - x₀

训练目标:
  L = E_{t, x₀, x₁} [ ||v_θ(x_t, t) - (x₁ - x₀)||² ]

采样 (ODE 积分):
  x₁ = x₀ + ∫₀¹ v_θ(x_t, t) dt
  ≈ x₀ + Σᵢ v_θ(x_tᵢ, tᵢ) · Δt  (Euler 方法)
```

**完整实现**：

```python
class FlowMatchingModel(nn.Module):
    """Flow Matching 时序预测模型"""

    def __init__(self, n_vars, seq_len, pred_len, d_model,
                 dim=256, n_layers=6, sigma_min=0.001):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.sigma_min = sigma_min

        # iTransformer backbone
        self.backbone = iTransformerEncoder(seq_len, n_vars, d_model)

        # Velocity network (DiT 或 UNet)
        self.velocity_net = DiTDenoiser(n_vars, pred_len, d_model, dim, n_layers)

    def get_interpolation(self, x0, x1, t):
        """
        计算最优传输插值路径

        Args:
            x0: [B, N, T] 噪声样本
            x1: [B, N, T] 数据样本
            t: [B] 时间 (0=噪声, 1=数据)
        Returns:
            x_t: [B, N, T] 插值点
            target_v: [B, N, T] 目标 velocity
        """
        t = t[:, None, None]  # [B, 1, 1]

        # 线性插值 (最优传输路径)
        x_t = (1 - t) * x0 + t * x1

        # 目标 velocity (直线方向)
        target_v = x1 - x0

        return x_t, target_v

    def forward_loss(self, x_enc, y_true):
        """
        训练损失计算

        Args:
            x_enc: [B, seq_len, N] 历史输入
            y_true: [B, pred_len, N] 目标
        """
        B = x_enc.shape[0]
        device = x_enc.device

        # Backbone: 提取条件特征
        z = self.backbone(x_enc)  # [B, N, d_model]

        # 归一化目标
        y_norm = self.normalize(y_true).permute(0, 2, 1)  # [B, N, T]

        # 采样噪声
        x0 = torch.randn_like(y_norm)

        # 采样时间 t ∈ (0, 1)
        t = torch.rand(B, device=device)

        # 计算插值和目标 velocity
        x_t, target_v = self.get_interpolation(x0, y_norm, t)

        # 预测 velocity
        pred_v = self.velocity_net(x_t, t, z)

        # 损失: 匹配 velocity
        loss = F.mse_loss(pred_v, target_v)

        return loss, {'loss_flow': loss.item()}

    @torch.no_grad()
    def sample(self, x_enc, n_samples=1, steps=50, method='euler'):
        """
        ODE 采样

        Args:
            x_enc: [B, seq_len, N] 历史输入
            n_samples: 采样数量
            steps: ODE 积分步数
            method: 'euler' 或 'heun' (2阶)
        """
        B = x_enc.shape[0]
        device = x_enc.device

        # Backbone
        z = self.backbone(x_enc)

        # 扩展 z 用于多样本
        z_exp = z.unsqueeze(0).expand(n_samples, -1, -1, -1)
        z_exp = z_exp.reshape(n_samples * B, *z.shape[1:])

        # 从噪声开始 (t=0)
        x = torch.randn(n_samples * B, self.n_vars, self.pred_len, device=device)

        # ODE 积分: 从 t=0 积分到 t=1
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n_samples * B,), i * dt, device=device)

            if method == 'euler':
                # Euler 方法
                v = self.velocity_net(x, t, z_exp)
                x = x + v * dt

            elif method == 'heun':
                # Heun 方法 (2阶 Runge-Kutta)
                v1 = self.velocity_net(x, t, z_exp)
                x_mid = x + v1 * dt
                t_next = torch.full((n_samples * B,), (i + 1) * dt, device=device)
                v2 = self.velocity_net(x_mid, t_next, z_exp)
                x = x + 0.5 * (v1 + v2) * dt

        # 反归一化
        x = x.reshape(n_samples, B, self.n_vars, self.pred_len)
        x = x.permute(0, 1, 3, 2)  # [n_samples, B, T, N]
        x = self.denormalize(x)

        return x.mean(dim=0), x.std(dim=0), x

    def normalize(self, y):
        """Instance normalization"""
        mean = y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True) + 1e-5
        return (y - mean) / std

    def denormalize(self, y, mean, std):
        return y * std + mean
```

**条件 Flow Matching (Conditional FM)**:

```python
class ConditionalFlowMatching(FlowMatchingModel):
    """条件 Flow Matching: 将确定性预测作为 x₁ 的先验"""

    def get_interpolation(self, x0, x1, t, y_det=None):
        """
        条件插值: 在确定性预测附近加噪

        Args:
            x0: [B, N, T] 噪声
            x1: [B, N, T] 真实目标
            t: [B] 时间
            y_det: [B, N, T] 确定性预测 (先验)
        """
        t = t[:, None, None]

        if y_det is not None:
            # 条件 FM: 以确定性预测为中心
            # x_t = (1-t) * N(y_det, σ) + t * x1
            sigma = 0.1 * (1 - t)  # 噪声随 t 增大减小
            x0_cond = y_det + sigma * torch.randn_like(y_det)
            x_t = (1 - t) * x0_cond + t * x1
            target_v = x1 - x0_cond
        else:
            x_t = (1 - t) * x0 + t * x1
            target_v = x1 - x0

        return x_t, target_v
```

**实施步骤**：

1. **创建 `models/FlowMatching.py`**
   - 实现基础 Flow Matching
   - 实现条件 Flow Matching

2. **创建 `exp/exp_flow_matching.py`**
   - 实现 Flow Matching 训练循环
   - 实现 ODE 采样评估

3. **添加命令行参数**
   - `--flow_steps`: ODE 积分步数
   - `--flow_method`: euler / heun

**预期收益**：
- 采样步数: 1000 → 50 (-95%)
- 生成质量: 与 DDPM 相当或更好
- 训练更稳定

---

#### 方案 F: Consistency Models

**核心思想**：学习从任意噪声点一步到达数据点的映射

**自洽性约束**：
```
对于同一条扩散轨迹上的任意两点 x_t 和 x_s (t ≠ s):
  f_θ(x_t, t) = f_θ(x_s, s) = x₀

即: 无论从轨迹哪个点出发，都应该映射到同一终点
```

**训练方式**：

```python
class ConsistencyModel(nn.Module):
    """Consistency Model for Time Series"""

    def __init__(self, ...):
        super().__init__()
        self.backbone = iTransformerEncoder(...)
        self.consistency_net = DiTDenoiser(...)

        # EMA 网络 (用于自洽性目标)
        self.ema_net = copy.deepcopy(self.consistency_net)
        for p in self.ema_net.parameters():
            p.requires_grad = False

    def consistency_function(self, x_t, t, z):
        """
        Consistency function: 从 x_t 预测 x₀

        在 t=0 时，应返回输入本身 (边界条件)
        """
        if t.min() < 0.01:
            # 边界条件: f(x_0, 0) = x_0
            return x_t

        # 否则，用网络预测
        return self.consistency_net(x_t, t, z)

    def forward_loss(self, x_enc, y_true):
        """Consistency Training Loss"""
        B = x_enc.shape[0]
        device = x_enc.device

        z = self.backbone(x_enc)
        y_norm = self.normalize(y_true).permute(0, 2, 1)

        # 采样相邻时间步 t 和 t + Δt
        t = torch.rand(B, device=device) * 0.99 + 0.01  # t ∈ (0.01, 1)
        delta_t = 0.01  # 小步长
        t_next = torch.clamp(t + delta_t, max=1.0)

        # 从 y_true 加噪到 t 和 t_next
        noise = torch.randn_like(y_norm)
        x_t = self.add_noise(y_norm, t, noise)
        x_t_next = self.add_noise(y_norm, t_next, noise)

        # 当前网络预测
        pred_t = self.consistency_function(x_t, t, z)

        # EMA 网络预测 (作为目标，stop gradient)
        with torch.no_grad():
            target = self.ema_consistency_function(x_t_next, t_next, z)

        # 自洽性损失: 同一轨迹上的点应映射到同一终点
        loss = F.mse_loss(pred_t, target)

        # 更新 EMA
        self.update_ema()

        return loss, {'loss_consistency': loss.item()}

    @torch.no_grad()
    def sample_one_step(self, x_enc, n_samples=1):
        """一步生成！"""
        z = self.backbone(x_enc)

        # 从纯噪声开始 (t=1)
        x = torch.randn(n_samples, *y_shape, device=device)
        t = torch.ones(n_samples * B, device=device)

        # 一步到位
        x0_pred = self.consistency_function(x, t, z)

        return x0_pred

    @torch.no_grad()
    def sample_multi_step(self, x_enc, n_samples=1, steps=4):
        """多步精化 (可选)"""
        z = self.backbone(x_enc)

        x = torch.randn(n_samples, *y_shape, device=device)

        timesteps = torch.linspace(1, 0.01, steps + 1)
        for i in range(steps):
            t = torch.full((n_samples * B,), timesteps[i], device=device)

            # 预测 x₀
            x0_pred = self.consistency_function(x, t, z)

            # 如果不是最后一步，加回部分噪声到下一个时间步
            if i < steps - 1:
                t_next = timesteps[i + 1]
                x = self.add_noise(x0_pred, t_next, torch.randn_like(x))

        return x0_pred

    def update_ema(self, decay=0.999):
        with torch.no_grad():
            for p_ema, p in zip(self.ema_net.parameters(),
                                self.consistency_net.parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)
```

**实施步骤**：

1. **实现 Consistency Model 基础框架**
2. **实现 Consistency Training**
3. **实现一步/多步采样**

**预期收益**：
- 一步生成 (1000x 加速!)
- 4 步精化可达接近 DDPM 质量
- 推理延迟降到毫秒级

---

### 训练策略优化

#### 方案 G: 端到端联合训练 ⭐⭐⭐ 强烈推荐

**核心思想**：取消两阶段分离，从一开始就联合优化 backbone 和 diffusion

**当前问题**：

```python
# Stage 1: 只训练 backbone
train_stage1():
    loss = MSE(y_det, y_true)  # backbone 只学确定性预测

# Stage 2: 冻结 backbone，只训练 diffusion
train_stage2():
    model.freeze_encoder()  # 梯度断开！
    loss = diffusion_loss   # diffusion 无法优化 backbone 的特征表示
```

**改进方案**：

```python
class EndToEndTrainer:
    """端到端联合训练器"""

    def __init__(self, model, warmup_epochs=10, total_epochs=50):
        self.model = model
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_loss_weights(self, epoch):
        """课程学习: 逐渐从确定性过渡到扩散"""
        if epoch < self.warmup_epochs:
            # 前期: 以确定性预测为主
            alpha = 1.0 - epoch / self.warmup_epochs * 0.3  # 1.0 → 0.7
        else:
            # 后期: 以扩散为主
            alpha = 0.3  # 固定 30% MSE + 70% Diffusion
        return alpha, 1 - alpha

    def train_step(self, x_enc, y_true, epoch):
        # 前向传播 (backbone + diffusion，梯度连通)
        y_det, z, means, stdev = self.model.backbone_forward(x_enc)

        # 确定性损失
        loss_det = F.mse_loss(y_det, y_true)

        # 扩散损失 (z 参与，梯度可以回传到 backbone!)
        loss_diff = self.model.diffusion_loss(y_true, z, means, stdev)

        # 联合损失 (课程学习)
        alpha, beta = self.get_loss_weights(epoch)
        loss = alpha * loss_det + beta * loss_diff

        return loss, {
            'loss_total': loss.item(),
            'loss_det': loss_det.item(),
            'loss_diff': loss_diff.item(),
            'alpha': alpha
        }

    def train_epoch(self, dataloader, epoch):
        self.model.train()

        # 动态学习率
        if epoch < self.warmup_epochs:
            # Warmup: backbone 正常 lr，diffusion 小 lr
            lr_backbone = 1e-4
            lr_diffusion = 1e-5
        else:
            # 联合: 都用较小 lr
            lr_backbone = 1e-5
            lr_diffusion = 1e-4

        self.set_learning_rates(lr_backbone, lr_diffusion)

        for batch in dataloader:
            loss, log = self.train_step(*batch, epoch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
```

**优化器配置**：

```python
def configure_optimizers(self):
    """分组学习率"""
    backbone_params = list(self.model.enc_embedding.parameters()) + \
                      list(self.model.encoder.parameters()) + \
                      list(self.model.projection.parameters())

    diffusion_params = list(self.model.denoise_net.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 0.01},
        {'params': diffusion_params, 'lr': 1e-4, 'weight_decay': 0.01}
    ])

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=self.total_epochs, eta_min=1e-6
    )

    return optimizer, scheduler
```

**实施步骤**：

1. **修改 `exp/exp_diffusion_forecast.py`**
   - 移除 `train_stage1` 和 `train_stage2` 分离
   - 实现 `EndToEndTrainer`

2. **修改损失计算**
   - 确保 `z` 参与扩散损失计算且保留梯度
   - 实现课程学习权重调度

3. **配置分组优化器**

**预期收益**：
- Backbone 学习对扩散有利的特征
- 整体性能提升 15-25%
- 训练更稳定

---

#### 方案 H: 时序感知损失函数

**核心思想**：利用时序数据的结构特性设计损失函数

**组件**：

```python
class TimeSeriesAwareLoss(nn.Module):
    """时序感知复合损失函数"""

    def __init__(self, lambda_point=1.0, lambda_trend=0.1,
                 lambda_freq=0.1, lambda_corr=0.05):
        super().__init__()
        self.lambda_point = lambda_point
        self.lambda_trend = lambda_trend
        self.lambda_freq = lambda_freq
        self.lambda_corr = lambda_corr

    def point_loss(self, pred, target):
        """点级 MSE 损失"""
        return F.mse_loss(pred, target)

    def trend_loss(self, pred, target):
        """
        趋势损失: 一阶差分的 MSE
        捕捉时序的局部变化趋势
        """
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return F.mse_loss(pred_diff, target_diff)

    def frequency_loss(self, pred, target):
        """
        频域损失: FFT 幅度谱的 MSE
        捕捉周期性模式
        """
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)

        # 幅度谱
        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()

        return F.mse_loss(pred_mag, target_mag)

    def correlation_loss(self, pred, target):
        """
        相关性损失: 变量间相关矩阵的 MSE
        保持多变量间的相关结构
        """
        # 计算相关矩阵
        def compute_corr(x):
            # x: [B, T, N]
            B, T, N = x.shape
            x_centered = x - x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True) + 1e-5
            x_norm = x_centered / x_std
            corr = torch.bmm(x_norm.transpose(1, 2), x_norm) / T
            return corr

        pred_corr = compute_corr(pred)
        target_corr = compute_corr(target)

        return F.mse_loss(pred_corr, target_corr)

    def forward(self, pred, target):
        """
        计算总损失

        Args:
            pred: [B, pred_len, N] 预测
            target: [B, pred_len, N] 真实值
        """
        loss_point = self.point_loss(pred, target)
        loss_trend = self.trend_loss(pred, target)
        loss_freq = self.frequency_loss(pred, target)
        loss_corr = self.correlation_loss(pred, target)

        total = (self.lambda_point * loss_point +
                 self.lambda_trend * loss_trend +
                 self.lambda_freq * loss_freq +
                 self.lambda_corr * loss_corr)

        return total, {
            'loss_point': loss_point.item(),
            'loss_trend': loss_trend.item(),
            'loss_freq': loss_freq.item(),
            'loss_corr': loss_corr.item()
        }
```

**扩展: 概率损失 (用于扩散)**

```python
class ProbabilisticLoss(nn.Module):
    """概率预测损失"""

    def crps_loss(self, samples, target):
        """
        CRPS 损失的可微近似

        Args:
            samples: [n_samples, B, T, N] 采样
            target: [B, T, N] 真实值
        """
        n_samples = samples.shape[0]

        # 预测均值
        mean_pred = samples.mean(dim=0)

        # |y - ŷ| 项
        term1 = torch.abs(target - mean_pred).mean()

        # E[|y' - y''|] / 2 项 (样本间差异)
        # 使用采样近似
        idx1 = torch.randperm(n_samples)[:n_samples//2]
        idx2 = torch.randperm(n_samples)[:n_samples//2]
        term2 = torch.abs(samples[idx1] - samples[idx2]).mean() / 2

        return term1 - term2

    def calibration_loss(self, samples, target, quantiles=[0.1, 0.5, 0.9]):
        """
        校准损失: 预测分位数应包含正确比例的真实值
        """
        loss = 0
        for q in quantiles:
            q_pred = torch.quantile(samples, q, dim=0)
            # 真实值应有 q 比例小于 q_pred
            actual_below = (target < q_pred).float().mean()
            loss += (actual_below - q) ** 2

        return loss / len(quantiles)
```

**实施步骤**：

1. **创建 `utils/losses.py`**
2. **在训练中使用复合损失**
3. **调整损失权重超参数**

**预期收益**：
- 趋势预测改善
- 变量相关性保持
- CRPS 指标改善

---

#### 方案 I: 渐进式扩散训练

**核心思想**：从简单（低噪声）到困难（高噪声）渐进学习

```python
class ProgressiveTrainer:
    """渐进式扩散训练"""

    def __init__(self, max_timesteps=1000, initial_timesteps=100,
                 increase_per_epoch=50):
        self.max_T = max_timesteps
        self.current_T = initial_timesteps
        self.increase_per_epoch = increase_per_epoch

    def update_curriculum(self, epoch):
        """每个 epoch 增加难度"""
        self.current_T = min(
            self.max_T,
            self.initial_timesteps + epoch * self.increase_per_epoch
        )

    def sample_timestep(self, batch_size, device):
        """只在当前难度范围内采样"""
        return torch.randint(0, self.current_T, (batch_size,), device=device)

    def get_snr_weights(self, t):
        """
        信噪比加权: 给困难样本更高权重

        SNR(t) = ᾱ_t / (1 - ᾱ_t)
        权重 = 1 / (SNR + 1)  (低 SNR = 高噪声 = 高权重)
        """
        alpha_t = self.alpha_cumprods[t]
        snr = alpha_t / (1 - alpha_t + 1e-8)
        weights = 1.0 / (snr + 1.0)
        return weights
```

**实施步骤**：

1. 实现课程学习调度器
2. 实现 SNR 加权损失
3. 集成到训练循环

---

### 效率优化

#### 方案 J: 混合精度 + 梯度检查点

```python
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

class EfficientModel(nn.Module):
    def forward_with_efficiency(self, x, t, z):
        # 梯度检查点: 用计算换显存
        def dit_block_fn(h, z, cond):
            for block in self.dit_blocks:
                h = block(h, z, cond)
            return h

        with autocast(dtype=torch.float16):  # FP16 混合精度
            h = self.init_proj(x)
            cond = self.cond_proj(z, t)
            h = checkpoint(dit_block_fn, h, z, cond, use_reentrant=False)
            out = self.final_proj(h)

        return out

# 训练循环
scaler = GradScaler()
for batch in dataloader:
    with autocast():
        loss = model.forward_loss(*batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

#### 方案 K: 模型蒸馏

```python
class DiffusionDistillation:
    """将多步模型蒸馏到少步模型"""

    def __init__(self, teacher, student):
        self.teacher = teacher  # 1000 步
        self.student = student  # 50 步

    def distill_step(self, x_enc, y_true):
        # Teacher: 高质量但慢
        with torch.no_grad():
            teacher_samples = self.teacher.sample(x_enc, steps=1000)

        # Student: 快但需要学习
        student_samples = self.student.sample(x_enc, steps=50)

        # 蒸馏损失: 匹配 Teacher 输出
        loss = F.mse_loss(student_samples.mean(0), teacher_samples.mean(0))

        return loss
```

---

## 实施路线图

### Phase 1: 基础优化 (建议首先实施)

| 步骤 | 任务 | 修改文件 | 预期效果 |
|------|------|----------|----------|
| 1.1 | v-prediction 参数化 | `models/iTransformerDiffusionDirect.py` | MSE -15%, 稳定性↑ |
| 1.2 | 端到端联合训练 | `exp/exp_diffusion_forecast.py` | MSE -10%, 训练效率↑ |
| 1.3 | 时序感知损失 | `utils/losses.py` (新建) | CRPS -10% |
| 1.4 | 混合精度训练 | `exp/exp_diffusion_forecast.py` | 显存 -30%, 速度↑ |

**总预期效果**: MSE: 0.60 → 0.42-0.45, CRPS: 0.50 → 0.35-0.40

### Phase 2: 架构升级

| 步骤 | 任务 | 修改文件 | 预期效果 |
|------|------|----------|----------|
| 2.1 | DiT 替代 UNet | `layers/DiT_layers.py` (新建) | MSE -10%, 架构统一 |
| 2.2 | 层级条件注入 | `layers/Diffusion_layers.py` | 变量 MSE 平衡 |
| 2.3 | AdaLayerNorm | `layers/DiT_layers.py` | 条件注入效率↑ |

**总预期效果**: MSE: 0.42 → 0.38-0.40

### Phase 3: 前沿技术

| 步骤 | 任务 | 修改文件 | 预期效果 |
|------|------|----------|----------|
| 3.1 | Flow Matching | `models/FlowMatching.py` (新建) | 采样步数 1000→50 |
| 3.2 | Consistency Model | `models/ConsistencyModel.py` (新建) | 一步生成 |
| 3.3 | 潜在空间扩散 | `models/LatentDiffusion.py` (新建) | 计算量 -75% |

**总预期效果**: 推理速度 20-1000x 提升

---

## 优先级排序与建议

### 最高优先级 (立即实施) ⭐⭐⭐

1. **v-prediction 参数化** (方案 D)
   - 实施难度: 低
   - 代码改动: ~50 行
   - 收益: 显著提升稳定性和精度

2. **端到端联合训练** (方案 G)
   - 实施难度: 中
   - 代码改动: ~100 行
   - 收益: 根本性解决两阶段割裂问题

### 高优先级 (第二轮实施) ⭐⭐

3. **DiT 架构** (方案 A)
   - 实施难度: 中高
   - 代码改动: 新建文件 ~300 行
   - 收益: 架构统一，便于后续扩展

4. **Flow Matching** (方案 E)
   - 实施难度: 中
   - 代码改动: 新建文件 ~200 行
   - 收益: 采样速度提升 20x

### 中优先级 (可选) ⭐

5. **时序感知损失** (方案 H)
6. **层级条件注入** (方案 C)
7. **渐进式训练** (方案 I)

### 低优先级 (研究性)

8. **Consistency Models** (方案 F) - 需要更多研究
9. **潜在空间扩散** (方案 B) - 需要预训练 VAE

---

## 预期收益分析

### 性能改进预测

| 指标 | 当前 | Phase 1 后 | Phase 2 后 | Phase 3 后 |
|------|------|------------|------------|------------|
| MSE | 0.5995 | 0.42-0.45 | 0.38-0.40 | 0.38-0.40 |
| MAE | ~0.50 | ~0.40 | ~0.38 | ~0.38 |
| CRPS | 0.495 | 0.35-0.40 | 0.30-0.35 | 0.30-0.35 |
| Calib-50% | 0.49 | 0.48-0.52 | 0.48-0.52 | 0.48-0.52 |
| Calib-90% | 0.88 | 0.88-0.92 | 0.88-0.92 | 0.88-0.92 |

### 效率改进预测

| 指标 | 当前 | Phase 1 后 | Phase 2 后 | Phase 3 后 |
|------|------|------------|------------|------------|
| 采样步数 | 1000 | 1000 | 1000 | **50** |
| 推理时间/batch | ~10s | ~7s | ~7s | **0.5s** |
| 训练显存 | 8GB | 5.5GB | 6GB | 4GB |
| 训练时间 | 基准 | -20% | -10% | -30% |

---

## 风险与缓解措施

### 风险 1: v-prediction 可能不适合时序数据

**缓解**:
- 先在小数据集验证
- 保留 x₀-prediction 作为 fallback
- 可以尝试混合参数化

### 风险 2: DiT 架构计算量可能更大

**缓解**:
- 使用较小的 patch_size
- 减少 DiT 层数
- 使用 Flash Attention

### 风险 3: Flow Matching 训练不稳定

**缓解**:
- 使用条件 Flow Matching (以确定性预测为先验)
- 实现 σ_min 正则化
- 渐进式增加 ODE 积分步数

### 风险 4: 端到端训练梯度爆炸

**缓解**:
- 使用梯度裁剪 (max_norm=1.0)
- 学习率 warmup
- 层级学习率衰减

---

## 附录

### A. 实验配置模板

```bash
# Phase 1 实验
python run.py \
  --task_name diffusion_forecast \
  --model iTransformerDiffusionV \
  --parameterization v \
  --training_mode end_to_end \
  --loss_type timeseries_aware \
  --use_amp \
  --data ETTh1 \
  --seq_len 96 --pred_len 96 \
  --d_model 128 --d_ff 128 \
  --diffusion_steps 1000 \
  --train_epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4

# Phase 3 Flow Matching 实验
python run.py \
  --task_name flow_matching_forecast \
  --model FlowMatchingTS \
  --flow_steps 50 \
  --flow_method heun \
  --data ETTh1 \
  --seq_len 96 --pred_len 96
```

### B. 参考文献

1. DiT: Peebles & Xie (2023). "Scalable Diffusion Models with Transformers"
2. Flow Matching: Lipman et al. (2023). "Flow Matching for Generative Modeling"
3. Consistency Models: Song et al. (2023). "Consistency Models"
4. v-prediction: Salimans & Ho (2022). "Progressive Distillation for Fast Sampling"
5. iTransformer: Liu et al. (2024). "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"

---

**文档版本**: v1.0
**最后更新**: 2025-01-20
**作者**: Claude Code (AI Assistant)
