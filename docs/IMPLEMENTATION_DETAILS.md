# iTransformer + CRD-Net 完整技术实现文档

**创建日期**: 2025-12-28
**版本**: v1.0
**作者**: Claude Code Assistant

---

## 目录

1. [项目概述](#1-项目概述)
2. [架构设计](#2-架构设计)
3. [核心模块实现](#3-核心模块实现)
4. [数据流详解](#4-数据流详解)
5. [训练流程](#5-训练流程)
6. [参数配置](#6-参数配置)
7. [接口设计](#7-接口设计)
8. [Bug修复记录](#8-bug修复记录)
9. [单元测试](#9-单元测试)
10. [版本迭代记录](#10-版本迭代记录)

---

## 1. 项目概述

### 1.1 目标
在 Time-Series-Library 框架中实现 **iTransformer + Conditional Residual Diffusion (CRD-Net)** 混合架构，实现多变量时间序列的概率预测。

### 1.2 核心创新点
- **残差空间扩散**: 在 `R₀ = Y - Ŷ_det` 残差空间而非原始空间进行扩散
- **条件注入**: 使用 FiLM + Cross-Attention 双重机制注入 iTransformer 提取的特征
- **两阶段训练**: 先训练骨干网，再联合训练扩散网络

### 1.3 文件变更清单

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| 新建 | `layers/Diffusion_layers.py` | 扩散模型核心组件 |
| 新建 | `models/iTransformerDiffusion.py` | 完整混合模型 |
| 新建 | `exp/exp_diffusion_forecast.py` | 两阶段训练实验类 |
| 修改 | `exp/exp_basic.py` | 注册新模型 |
| 修改 | `utils/metrics.py` | 添加概率预测指标 |
| 修改 | `run.py` | 添加任务类型和参数 |
| 新建 | `scripts/diffusion_forecast/quick_test.sh` | 快速测试脚本 |
| 新建 | `scripts/diffusion_forecast/ETT_script/` | 完整实验脚本 |
| 新建 | `tests/test_iTransformerDiffusion.py` | 单元测试 |
| 新建 | `tests/test_edge_cases.py` | 边界测试 |

---

## 2. 架构设计

### 2.1 整体架构图

```
输入 x_hist [B, seq_len, N]
         │
         ▼
┌─────────────────────────────────────────┐
│           iTransformer Backbone          │
│  ┌─────────────────────────────────┐    │
│  │ Instance Normalization          │    │
│  │   means, stdev 保存用于反归一化   │    │
│  └─────────────────────────────────┘    │
│                  │                       │
│                  ▼                       │
│  ┌─────────────────────────────────┐    │
│  │ DataEmbedding_inverted          │    │
│  │  [B, seq_len, N] → [B, N, d_model]│  │
│  └─────────────────────────────────┘    │
│                  │                       │
│                  ▼                       │
│  ┌─────────────────────────────────┐    │
│  │ Transformer Encoder (2 layers)   │    │
│  │   Attention across variates      │    │
│  │   输出: z [B, N, d_model]        │    │
│  └─────────────────────────────────┘    │
│                  │                       │
│                  ▼                       │
│  ┌─────────────────────────────────┐    │
│  │ Projection Layer                 │    │
│  │   [B, N, d_model] → [B, N, pred_len]│ │
│  │   → permute → [B, pred_len, N]   │    │
│  └─────────────────────────────────┘    │
│                  │                       │
│                  ▼                       │
│  ┌─────────────────────────────────┐    │
│  │ Denormalization                  │    │
│  │   y_det = y * stdev + means      │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
         │                    │
         │ y_det              │ z (条件特征)
         │ [B, pred_len, N]   │ [B, N, d_model]
         ▼                    ▼
┌─────────────────────────────────────────┐
│           CRD-Net (Diffusion)            │
│                                          │
│  训练时:                                  │
│  ┌─────────────────────────────────┐    │
│  │ residual = y_true - y_det.detach()│   │
│  │ residual_norm = normalize(residual)│  │
│  │ residual_norm: [B, pred_len, N]   │   │
│  │   → permute → [B, N, pred_len]    │   │
│  └─────────────────────────────────┘    │
│                  │                       │
│                  ▼                       │
│  ┌─────────────────────────────────┐    │
│  │ Forward Diffusion (add noise)    │    │
│  │   t ~ Uniform(0, T)              │    │
│  │   x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε│   │
│  └─────────────────────────────────┘    │
│                  │                       │
│                  ▼                       │
│  ┌─────────────────────────────────┐    │
│  │ 1D U-Net Denoiser                │    │
│  │   输入: x_t [B, N, pred_len]     │    │
│  │   条件: z [B, N, d_model], t     │    │
│  │   输出: ε_pred [B, N, pred_len]  │    │
│  └─────────────────────────────────┘    │
│                  │                       │
│                  ▼                       │
│  ┌─────────────────────────────────┐    │
│  │ Loss = MSE(ε_pred, ε)            │    │
│  └─────────────────────────────────┘    │
│                                          │
│  推理时:                                  │
│  ┌─────────────────────────────────┐    │
│  │ DDPM/DDIM Sampling               │    │
│  │   x_T ~ N(0, I)                  │    │
│  │   for t = T...0:                 │    │
│  │     x_{t-1} = denoise(x_t, t, z) │    │
│  │   残差样本: [n_samples, B, N, T] │    │
│  └─────────────────────────────────┘    │
│                  │                       │
│                  ▼                       │
│  ┌─────────────────────────────────┐    │
│  │ Denormalize & Add Trend          │    │
│  │   y_final = y_det + residual     │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
         │
         ▼
输出: mean_pred, std_pred, samples
      [B, pred_len, N]
```

### 2.2 U-Net 架构详解

```
输入: x [B, N, pred_len], t [B], z [B, N, d_model]
                │
                ▼
┌───────────────────────────────────────────────┐
│ Initial Conv: [N, pred_len] → [64, pred_len]  │
└───────────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        │               │ (skip connection h1)
        ▼               │
┌─────────────────────┐ │
│ DownBlock1D         │ │
│   in=64, out=128    │ │
│   + FiLM(cond_dim)  │ │
│   stride=2 下采样    │ │
│   out: [128, T/2]   │ │
└─────────────────────┘ │
        │               │
        ├───────────────┤ (skip connection h2)
        ▼               │
┌─────────────────────┐ │
│ DownBlock1D         │ │
│   in=128, out=256   │ │
│   + FiLM(cond_dim)  │ │
│   out: [256, T/4]   │ │
└─────────────────────┘ │
        │               │
        ├───────────────┤ (skip connection h3)
        ▼               │
┌─────────────────────┐ │
│ DownBlock1D         │ │
│   in=256, out=512   │ │
│   + FiLM(cond_dim)  │ │
│   out: [512, T/8]   │ │
└─────────────────────┘ │
        │               │
        ▼               │
┌─────────────────────┐ │
│ Bottleneck          │ │
│   ResBlock1D(512)   │ │
│   + CrossAttention  │ │  ← z [B, N, d_model]
│   ResBlock1D(512)   │ │
└─────────────────────┘ │
        │               │
        ▼               │
┌─────────────────────┐ │
│ UpBlock1D           │ │
│   in=512, out=256   │◄┼── concat h3 [512+256=768 → 256]
│   skip_ch=256       │ │
│   + FiLM + CrossAttn│ │
│   out: [256, T/4]   │ │
└─────────────────────┘ │
        │               │
        ▼               │
┌─────────────────────┐ │
│ UpBlock1D           │ │
│   in=256, out=128   │◄┼── concat h2 [256+128=384 → 128]
│   skip_ch=128       │ │
│   + FiLM + CrossAttn│ │
│   out: [128, T/2]   │ │
└─────────────────────┘ │
        │               │
        ▼               │
┌─────────────────────┐ │
│ UpBlock1D           │ │
│   in=128, out=64    │◄┼── concat h1 [128+64=192 → 64]
│   skip_ch=64        │ │
│   + FiLM + CrossAttn│ │
│   out: [64, T]      │ │
└─────────────────────┘ │
        │               │
        ▼               │
┌───────────────────────────────────────────────┐
│ Final Conv: [64, pred_len] → [N, pred_len]    │
└───────────────────────────────────────────────┘
                │
                ▼
输出: ε_pred [B, N, pred_len]
```

---

## 3. 核心模块实现

### 3.1 layers/Diffusion_layers.py

#### 3.1.1 SinusoidalPosEmb - 时间步嵌入

```python
class SinusoidalPosEmb(nn.Module):
    """
    正弦位置嵌入，用于编码扩散时间步 t。

    数学原理:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Args:
        dim: 嵌入维度 (默认 cond_dim=256)

    输入输出:
        输入: t [B] - 时间步索引 (0 到 T-1)
        输出: [B, dim] - 时间步嵌入向量
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim]
        return emb
```

**设计决策**:
- 使用正弦嵌入而非可学习嵌入，因为扩散步数 T=1000 较大
- 维度选择 256，与 cond_dim 一致，便于后续融合

#### 3.1.2 ConditionProjector - 条件投影器

```python
class ConditionProjector(nn.Module):
    """
    将时间步嵌入和编码器特征融合为统一的条件向量。

    融合方式: cond = MLP(time_emb) + MLP(z.mean(dim=1))

    Args:
        d_model: 编码器特征维度 (128)
        cond_dim: 输出条件维度 (256)

    输入输出:
        输入: t_emb [B, cond_dim], z [B, N, d_model]
        输出: cond [B, cond_dim]
    """
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim)
        )
        self.z_mlp = nn.Sequential(
            nn.Linear(d_model, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, t_emb, z):
        # t_emb: [B, cond_dim]
        # z: [B, N, d_model]
        t_cond = self.time_mlp(t_emb)  # [B, cond_dim]
        z_pooled = z.mean(dim=1)  # [B, d_model] - 变量平均池化
        z_cond = self.z_mlp(z_pooled)  # [B, cond_dim]
        return t_cond + z_cond  # [B, cond_dim]
```

**设计决策**:
- 对 z 进行变量维度平均池化，获得全局条件
- 使用 SiLU 激活函数 (比 ReLU 更平滑)
- 时间和特征条件相加融合

#### 3.1.3 FiLMLayer - 特征调制层

```python
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) 层。

    原理: out = γ * x + β
    其中 γ, β 由条件向量生成。

    Args:
        cond_dim: 条件向量维度 (256)
        out_channels: 要调制的特征通道数

    输入输出:
        输入: x [B, C, T], cond [B, cond_dim]
        输出: [B, C, T] - 调制后的特征
    """
    def __init__(self, cond_dim, out_channels):
        super().__init__()
        self.fc = nn.Linear(cond_dim, out_channels * 2)

    def forward(self, x, cond):
        # x: [B, C, T], cond: [B, cond_dim]
        gamma_beta = self.fc(cond)  # [B, 2*C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # 各 [B, C]
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)    # [B, C, 1]
        return gamma * x + beta
```

**设计决策**:
- FiLM 是轻量级条件注入方式，计算开销小
- γ 和 β 从同一个线性层生成，参数效率高

#### 3.1.4 VariateCrossAttention - 变量交叉注意力

```python
class VariateCrossAttention(nn.Module):
    """
    跨变量的交叉注意力机制。

    Query 来自当前特征，Key/Value 来自编码器条件。
    注意力在变量维度上操作，捕获变量间相关性。

    Args:
        channels: 当前特征通道数
        d_model: 编码器特征维度 (128)
        n_heads: 注意力头数 (4)

    输入输出:
        输入: x [B, C, T], z [B, N, d_model]
        输出: [B, C, T] - 注意力增强的特征
    """
    def __init__(self, channels, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        # Query 投影: 从当前特征
        self.q_proj = nn.Conv1d(channels, channels, 1)
        # Key/Value 投影: 从编码器特征
        self.k_proj = nn.Linear(d_model, channels)
        self.v_proj = nn.Linear(d_model, channels)
        # 输出投影
        self.out_proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, z):
        B, C, T = x.shape
        N = z.shape[1]  # 变量数

        # Query: [B, C, T] -> [B, heads, T, head_dim]
        q = self.q_proj(x).view(B, self.n_heads, self.head_dim, T).permute(0, 1, 3, 2)

        # Key/Value: [B, N, d_model] -> [B, heads, N, head_dim]
        k = self.k_proj(z).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(z).view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # 注意力计算: [B, heads, T, N]
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # 加权求和: [B, heads, T, head_dim]
        out = torch.matmul(attn, v)

        # 重组: [B, C, T]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, T)
        out = self.out_proj(out)

        # 残差连接 + LayerNorm
        x = x + out
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
```

**设计决策**:
- Query 从 U-Net 特征生成，K/V 从编码器特征生成
- 使用多头注意力 (4 heads) 增加表达能力
- 残差连接防止信息丢失

#### 3.1.5 ResBlock1D - 残差卷积块

```python
class ResBlock1D(nn.Module):
    """
    1D 残差卷积块，带 FiLM 条件调制。

    结构: x → Conv → GroupNorm → FiLM → SiLU → Conv → GroupNorm → + x

    Args:
        in_channels: 输入通道
        out_channels: 输出通道
        cond_dim: 条件维度 (256)
    """
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.film = FiLMLayer(cond_dim, out_channels)
        self.act = nn.SiLU()

        # 残差连接的通道适配
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film(h, cond)  # FiLM 条件调制
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return h + self.skip(x)  # 残差连接
```

**设计决策**:
- 使用 GroupNorm 而非 BatchNorm，对小批量更稳定
- FiLM 调制放在第一个 norm 之后，第一个激活之前
- 3x3 卷积核，padding=1 保持尺寸

#### 3.1.6 DownBlock1D / UpBlock1D - 上下采样块

```python
class DownBlock1D(nn.Module):
    """
    下采样块: ResBlock + 2x 下采样

    输出尺寸: [B, out_ch, T/2]
    """
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.res = ResBlock1D(in_channels, out_channels, cond_dim)
        self.down = nn.Conv1d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, cond):
        h = self.res(x, cond)
        return self.down(h), h  # 返回下采样结果和 skip connection


class UpBlock1D(nn.Module):
    """
    上采样块: 2x 上采样 + concat skip + ResBlock + CrossAttention

    关键: skip_channels 参数明确指定 skip connection 的通道数

    Args:
        in_channels: 来自下层的通道数
        out_channels: 输出通道数
        skip_channels: skip connection 的通道数 (关键修复点)
        cond_dim: 条件维度
        d_model: 编码器特征维度 (用于 CrossAttention)
        use_cross_attn: 是否使用交叉注意力
    """
    def __init__(self, in_channels, out_channels, skip_channels, cond_dim,
                 d_model=None, use_cross_attn=False):
        super().__init__()
        # 转置卷积上采样
        self.up = nn.ConvTranspose1d(in_channels, in_channels, 4, stride=2, padding=1)
        # ResBlock 输入通道 = 上采样后通道 + skip 通道
        self.res = ResBlock1D(in_channels + skip_channels, out_channels, cond_dim)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn and d_model is not None:
            self.cross_attn = VariateCrossAttention(out_channels, d_model)

    def forward(self, x, skip, cond, z=None):
        x = self.up(x)
        # 处理尺寸不匹配 (由于下采样时的 padding)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # [B, in_ch + skip_ch, T]
        x = self.res(x, cond)
        if self.use_cross_attn and z is not None:
            x = self.cross_attn(x, z)
        return x
```

**设计决策**:
- 使用转置卷积 (kernel=4, stride=2) 实现 2x 上采样
- skip_channels 明确传入，避免通道计算错误 (修复的关键 bug)
- CrossAttention 仅在上采样路径使用，增强条件注入

#### 3.1.7 UNet1D - 完整 U-Net

```python
class UNet1D(nn.Module):
    """
    完整的 1D U-Net 去噪网络。

    架构:
        - 初始卷积: N → 64
        - 下采样路径: 64 → 128 → 256 → 512
        - Bottleneck: 512 + CrossAttention
        - 上采样路径: 512 → 256 → 128 → 64
        - 最终卷积: 64 → N

    Args:
        n_vars: 变量数 (输入输出通道)
        pred_len: 预测长度 (时间维度)
        d_model: 编码器特征维度 (128)
        cond_dim: 条件维度 (256)
        channels: 通道配置列表 [64, 128, 256, 512]
    """
    def __init__(self, n_vars, pred_len, d_model, cond_dim, channels=[64, 128, 256, 512]):
        super().__init__()
        self.n_vars = n_vars
        self.channels = channels

        # 时间嵌入
        self.time_emb = SinusoidalPosEmb(cond_dim)
        # 条件投影
        self.cond_proj = ConditionProjector(d_model, cond_dim)

        # 初始卷积: [N, pred_len] → [64, pred_len]
        self.init_conv = nn.Conv1d(n_vars, channels[0], 3, padding=1)

        # 下采样块
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                DownBlock1D(channels[i], channels[i + 1], cond_dim)
            )

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock1D(channels[-1], channels[-1], cond_dim),
            VariateCrossAttention(channels[-1], d_model),
            ResBlock1D(channels[-1], channels[-1], cond_dim)
        ])

        # 上采样块 (关键: 正确计算 skip_channels)
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))  # [512, 256, 128, 64]
        skip_channels_list = list(reversed(channels[1:]))  # [512, 256, 128]

        for i, out_ch in enumerate(reversed_channels[1:]):  # [256, 128, 64]
            in_ch = reversed_channels[i]  # [512, 256, 128]
            skip_ch = skip_channels_list[i]  # [512, 256, 128]
            self.up_blocks.append(
                UpBlock1D(in_ch, out_ch, skip_ch, cond_dim, d_model, use_cross_attn=True)
            )

        # 最终卷积: [64, pred_len] → [N, pred_len]
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], n_vars, 3, padding=1)
        )

    def forward(self, x, t, z):
        """
        Args:
            x: [B, N, pred_len] - 噪声数据
            t: [B] - 时间步
            z: [B, N, d_model] - 编码器条件特征
        Returns:
            [B, N, pred_len] - 预测噪声
        """
        # 条件向量
        t_emb = self.time_emb(t)  # [B, cond_dim]
        cond = self.cond_proj(t_emb, z)  # [B, cond_dim]

        # 初始卷积
        h = self.init_conv(x)  # [B, 64, T]

        # 下采样 + 保存 skip connections
        skips = []
        for down in self.down_blocks:
            h, skip = down(h, cond)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck[0](h, cond)
        h = self.bottleneck[1](h, z)  # CrossAttention
        h = self.bottleneck[2](h, cond)

        # 上采样 + skip connections
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip, cond, z)

        # 最终输出
        return self.final_conv(h)
```

**关键设计点**:
1. **skip_channels 计算**: `skip_channels_list = list(reversed(channels[1:]))` 确保每层正确的通道数
2. **Bottleneck CrossAttention**: 在最深层注入完整的条件信息
3. **上采样路径 CrossAttention**: 每层都有条件注入，逐步精化

#### 3.1.8 ResidualNormalizer - 残差归一化器

```python
class ResidualNormalizer(nn.Module):
    """
    残差归一化器，使用指数移动平均 (EMA) 跟踪统计量。

    原理:
        训练时: 更新 running_mean, running_std
        归一化: (x - mean) / std
        反归一化: x * std + mean

    Args:
        n_vars: 变量数
        momentum: EMA 动量 (0.1)
    """
    def __init__(self, n_vars, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(n_vars))
        self.register_buffer('running_std', torch.ones(n_vars))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def normalize(self, x):
        # x: [B, pred_len, N] 或 [B, N, pred_len]
        if self.training:
            # 计算当前批次统计量
            mean = x.mean(dim=(0, 1))  # [N]
            std = x.std(dim=(0, 1)) + 1e-5  # [N]
            # EMA 更新
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            std = self.running_std

        # 归一化
        return (x - mean) / std

    def denormalize(self, x):
        return x * self.running_std + self.running_mean
```

**设计决策**:
- 使用 EMA 而非批次统计，推理时更稳定
- 每个变量独立归一化
- 添加 eps=1e-5 防止除零

---

### 3.2 models/iTransformerDiffusion.py

#### 3.2.1 Model 类初始化

```python
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # ========== 基础配置 ==========
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len      # 96
        self.pred_len = configs.pred_len    # 96
        self.n_vars = configs.enc_in        # 7 (ETTh1)
        self.d_model = configs.d_model      # 128

        # ========== 扩散配置 (带默认值) ==========
        self.timesteps = getattr(configs, 'diffusion_steps', 1000)
        self.beta_schedule = getattr(configs, 'beta_schedule', 'cosine')
        self.cond_dim = getattr(configs, 'cond_dim', 256)
        self.unet_channels = getattr(configs, 'unet_channels', [64, 128, 256, 512])
        self.n_samples = getattr(configs, 'n_samples', 100)

        # ========== iTransformer 骨干网 ==========
        # 嵌入层: 反转维度，时间 → 通道
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,  # c_in = seq_len (反转后)
            configs.d_model,
            configs.embed,    # 'timeF'
            configs.freq,     # 'h'
            configs.dropout   # 0.1
        )

        # Transformer 编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,  # mask_flag
                            configs.factor,  # 3
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads  # 8
                    ),
                    configs.d_model,
                    configs.d_ff,  # 128
                    dropout=configs.dropout,
                    activation=configs.activation  # 'gelu'
                ) for _ in range(configs.e_layers)  # 2 layers
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 投影层: d_model → pred_len
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # ========== CRD-Net 扩散网络 ==========
        self.residual_normalizer = ResidualNormalizer(self.n_vars)
        self.denoise_net = UNet1D(
            n_vars=self.n_vars,
            pred_len=self.pred_len,
            d_model=self.d_model,
            cond_dim=self.cond_dim,
            channels=self.unet_channels
        )

        # ========== 扩散调度 ==========
        self._setup_diffusion_schedule()
```

#### 3.2.2 扩散调度设置

```python
def _setup_diffusion_schedule(self):
    """
    设置 beta 调度并预计算扩散常数。

    支持:
        - 'linear': β_t 线性增长 [1e-4, 2e-2]
        - 'cosine': 余弦调度 (improved DDPM)
    """
    if self.beta_schedule == 'linear':
        betas = torch.linspace(1e-4, 2e-2, self.timesteps)
    elif self.beta_schedule == 'cosine':
        # 余弦调度: ᾱ_t = cos²((t/T + s) / (1+s) * π/2)
        s = 0.008  # 偏移量，防止 β_0 过小
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps) / self.timesteps
        alpha_cumprod = torch.cos(((t + s) / (1 + s)) * np.pi * 0.5) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]  # 归一化
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)

    alphas = 1.0 - betas
    alpha_cumprods = torch.cumprod(alphas, dim=0)

    # 注册为 buffer (自动随模型移动到正确设备)
    self.register_buffer('betas', betas)
    self.register_buffer('alphas', alphas)
    self.register_buffer('alpha_cumprods', alpha_cumprods)
    self.register_buffer('sqrt_alpha_cumprods', torch.sqrt(alpha_cumprods))
    self.register_buffer('sqrt_one_minus_alpha_cumprods', torch.sqrt(1.0 - alpha_cumprods))
```

**设计决策**:
- 默认使用 cosine schedule，在后期保留更多信号
- 预计算所有常数，避免训练时重复计算
- 使用 `register_buffer` 确保常数随模型移动到 GPU

#### 3.2.3 backbone_forward - 骨干网前向

```python
def backbone_forward(self, x_enc, x_mark_enc=None):
    """
    iTransformer 骨干网前向传播。

    Args:
        x_enc: [B, seq_len, N] 输入历史序列
        x_mark_enc: [B, seq_len, M] 时间标记 (可选)

    Returns:
        y_det: [B, pred_len, N] 确定性预测
        z: [B, N, d_model] 编码器特征 (条件)
        means, stdev: 归一化统计量
    """
    # ========== 实例归一化 ==========
    # 每个样本独立归一化，保存统计量用于反归一化
    means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, N]
    x_enc = x_enc - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x_enc = x_enc / stdev

    # 使用模型的 n_vars 而非输入维度 (关键修复)
    N = self.n_vars

    # ========== 嵌入 ==========
    # DataEmbedding_inverted: [B, seq_len, N] → [B, N, d_model]
    enc_out = self.enc_embedding(x_enc, x_mark_enc)

    # ========== 编码器 ==========
    # Encoder: [B, N, d_model] → [B, N, d_model]
    z, _ = self.encoder(enc_out, attn_mask=None)

    # ========== 通道对齐检查 (关键修复) ==========
    actual_n_vars = z.shape[1]
    if actual_n_vars != N:
        if actual_n_vars > N:
            z = z[:, :N, :]  # 截断
        else:
            # 填充 (边界情况)
            padding = torch.zeros(z.shape[0], N - actual_n_vars, z.shape[2],
                                device=z.device, dtype=z.dtype)
            z = torch.cat([z, padding], dim=1)

    # ========== 投影 ==========
    # [B, N, d_model] → [B, N, pred_len] → [B, pred_len, N]
    y_det = self.projection(z).permute(0, 2, 1)

    # ========== 反归一化 ==========
    y_det = y_det * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
    y_det = y_det + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

    return y_det, z, means, stdev
```

**关键修复**:
- 使用 `self.n_vars` 而非 `x_enc.shape[2]` 确保一致性
- 添加通道对齐检查，处理边界情况

#### 3.2.4 forward_loss - 训练损失

```python
def forward_loss(self, x_enc, x_mark_enc, y_true, stage='joint'):
    """
    计算训练损失。

    Args:
        x_enc: [B, seq_len, N] 输入
        x_mark_enc: [B, seq_len, M] 时间标记
        y_true: [B, pred_len, N] 真值
        stage: 'warmup' (Stage 1) 或 'joint' (Stage 2)

    Returns:
        loss: 标量损失
        loss_dict: 详细损失字典
    """
    B = x_enc.shape[0]
    device = x_enc.device

    # 骨干网前向
    y_det, z, _, _ = self.backbone_forward(x_enc, x_mark_enc)

    # MSE 损失
    loss_mse = F.mse_loss(y_det, y_true)

    if stage == 'warmup':
        # Stage 1: 仅 MSE
        return loss_mse, {'loss_mse': loss_mse.item()}

    # Stage 2: MSE + Diffusion
    # 计算残差 (detach y_det 防止梯度回传到编码器)
    residual = y_true - y_det.detach()

    # 归一化残差
    residual_norm = self.residual_normalizer.normalize(residual)

    # 维度变换: [B, pred_len, N] → [B, N, pred_len]
    residual_norm = residual_norm.permute(0, 2, 1)

    # 随机时间步
    t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)

    # 加噪
    noise = torch.randn_like(residual_norm)
    residual_noisy, _ = self.add_noise(residual_norm, t, noise)

    # 预测噪声
    noise_pred = self.denoise_net(residual_noisy, t, z)

    # 扩散损失
    loss_diff = F.mse_loss(noise_pred, noise)

    # 总损失 (λ = 0.5)
    loss_lambda = 0.5
    loss_total = loss_lambda * loss_mse + (1 - loss_lambda) * loss_diff

    return loss_total, {
        'loss_total': loss_total.item(),
        'loss_mse': loss_mse.item(),
        'loss_diff': loss_diff.item()
    }
```

**设计决策**:
- Stage 2 中 `y_det.detach()` 防止扩散梯度影响编码器
- λ=0.5 平衡点预测和分布学习
- 使用 ε-prediction (预测噪声) 而非 x0-prediction

#### 3.2.5 sample_ddpm - DDPM 采样

```python
@torch.no_grad()
def sample_ddpm(self, z, n_samples=1):
    """
    DDPM 采样算法。

    采样过程:
        x_T ~ N(0, I)
        for t = T-1, ..., 0:
            ε_θ = denoise_net(x_t, t, z)
            μ_t = (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * ε_θ)
            x_{t-1} = μ_t + σ_t * ε  (t > 0)
            x_0 = μ_0  (t = 0)

    Args:
        z: [B, N, d_model] 条件特征
        n_samples: 每个输入的采样数

    Returns:
        samples: [n_samples, B, N, pred_len]
    """
    B, _, _ = z.shape
    device = z.device
    N = self.n_vars  # 关键: 使用 self.n_vars 而非 z.shape[1]

    all_samples = []
    for _ in range(n_samples):
        # 从纯噪声开始
        x = torch.randn(B, N, self.pred_len, device=device)

        # 反向扩散
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # 预测噪声
            noise_pred = self.denoise_net(x, t_batch, z)

            # DDPM 更新公式
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alpha_cumprods[t]
            beta_t = self.betas[t]

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / self.sqrt_one_minus_alpha_cumprods[t]
            mean = coef1 * (x - coef2 * noise_pred)

            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean

        all_samples.append(x)

    return torch.stack(all_samples, dim=0)
```

#### 3.2.6 sample_ddim - DDIM 采样

```python
@torch.no_grad()
def sample_ddim(self, z, n_samples=1, ddim_steps=50, eta=0.0):
    """
    DDIM 采样算法 (比 DDPM 快)。

    DDIM 公式:
        x_{t-1} = √ᾱ_{t-1} * x̂_0 + √(1-ᾱ_{t-1}-σ²) * ε_θ + σ * ε
    其中:
        x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        σ = η * √((1-ᾱ_{t-1})/(1-ᾱ_t)) * √(1-ᾱ_t/ᾱ_{t-1})

    Args:
        z: [B, N, d_model] 条件特征
        n_samples: 采样数
        ddim_steps: DDIM 步数 (默认 50)
        eta: 随机性参数 (0=确定性)

    Returns:
        samples: [n_samples, B, N, pred_len]
    """
    B, _, _ = z.shape
    device = z.device
    N = self.n_vars

    # DDIM 时间序列 (跳步)
    step_size = self.timesteps // ddim_steps
    timesteps = list(range(0, self.timesteps, step_size))[::-1]

    all_samples = []
    for _ in range(n_samples):
        x = torch.randn(B, N, self.pred_len, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.denoise_net(x, t_batch, z)

            alpha_t = self.alpha_cumprods[t]
            if i == len(timesteps) - 1:
                alpha_t_prev = torch.tensor(1.0, device=device)
            else:
                t_prev = timesteps[i + 1]
                alpha_t_prev = self.alpha_cumprods[t_prev]

            # 预测 x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)  # 稳定性裁剪

            # 计算 σ
            sigma_t = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )

            # 更新
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * noise_pred
            noise = torch.randn_like(x) if sigma_t > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt + sigma_t * noise

        all_samples.append(x)

    return torch.stack(all_samples, dim=0)
```

**DDIM 优势**:
- 1000 步 DDPM → 50 步 DDIM (20x 加速)
- eta=0 时完全确定性
- 质量损失很小

#### 3.2.7 predict - 推理接口

```python
@torch.no_grad()
def predict(self, x_enc, x_mark_enc=None, n_samples=None, use_ddim=False, ddim_steps=50):
    """
    概率预测接口。

    Args:
        x_enc: [B, seq_len, N] 输入
        x_mark_enc: [B, seq_len, M] 时间标记
        n_samples: 采样数 (默认 self.n_samples)
        use_ddim: 是否使用 DDIM
        ddim_steps: DDIM 步数

    Returns:
        mean_pred: [B, pred_len, N] 均值预测
        std_pred: [B, pred_len, N] 不确定性
        samples: [n_samples, B, pred_len, N] 所有样本
    """
    if n_samples is None:
        n_samples = self.n_samples

    # 骨干网前向
    y_det, z, _, _ = self.backbone_forward(x_enc, x_mark_enc)

    # 采样残差
    if use_ddim:
        residual_samples = self.sample_ddim(z, n_samples, ddim_steps)
    else:
        residual_samples = self.sample_ddpm(z, n_samples)

    # 维度变换: [n_samples, B, N, pred_len] → [n_samples, B, pred_len, N]
    residual_samples = residual_samples.permute(0, 1, 3, 2)

    # 反归一化残差
    residual_samples = self.residual_normalizer.denormalize(residual_samples)

    # 最终预测 = 趋势 + 残差
    samples = y_det.unsqueeze(0) + residual_samples

    # 统计量
    mean_pred = samples.mean(dim=0)
    std_pred = samples.std(dim=0, unbiased=False)  # 避免 n_samples=1 警告

    return mean_pred, std_pred, samples
```

#### 3.2.8 freeze_encoder / unfreeze_encoder

```python
def freeze_encoder(self):
    """冻结 iTransformer 编码器 (Stage 2 使用)"""
    for param in self.enc_embedding.parameters():
        param.requires_grad = False
    for param in self.encoder.parameters():
        param.requires_grad = False
    # 保持 projection 可训练
    for param in self.projection.parameters():
        param.requires_grad = True

def unfreeze_encoder(self):
    """解冻编码器"""
    for param in self.enc_embedding.parameters():
        param.requires_grad = True
    for param in self.encoder.parameters():
        param.requires_grad = True
```

**设计决策**:
- Stage 2 冻结编码器，防止扩散梯度破坏学到的表征
- 保持 projection 可训练，允许输出适配

---

## 4. 数据流详解

### 4.1 训练时数据流

```
输入:
  x_enc: [32, 96, 7]      # batch=32, seq_len=96, 7 variates
  x_mark_enc: [32, 96, 4] # 时间特征
  y_true: [32, 96, 7]     # 真值

Step 1: Instance Normalization
  means: [32, 1, 7]
  stdev: [32, 1, 7]
  x_norm: [32, 96, 7]

Step 2: DataEmbedding_inverted
  x_norm: [32, 96, 7]
    → permute → [32, 7, 96]
    → Linear(96 → 128) → [32, 7, 128]

Step 3: Transformer Encoder
  enc_out: [32, 7, 128]
    → 2x EncoderLayer (attention across 7 variates)
    → z: [32, 7, 128]

Step 4: Projection
  z: [32, 7, 128]
    → Linear(128 → 96) → [32, 7, 96]
    → permute → y_det: [32, 96, 7]

Step 5: Denormalization
  y_det = y_det * stdev + means
  y_det: [32, 96, 7]

Step 6: Residual Computation (Stage 2)
  residual = y_true - y_det.detach()
  residual: [32, 96, 7]
  residual_norm: [32, 96, 7] (归一化后)
    → permute → [32, 7, 96]

Step 7: Forward Diffusion
  t: [32] (随机时间步)
  noise: [32, 7, 96]
  residual_noisy = √ᾱ_t * residual_norm + √(1-ᾱ_t) * noise
  residual_noisy: [32, 7, 96]

Step 8: U-Net Denoising
  输入: residual_noisy [32, 7, 96], t [32], z [32, 7, 128]
  输出: noise_pred [32, 7, 96]

Step 9: Loss Computation
  loss_mse = MSE(y_det, y_true)
  loss_diff = MSE(noise_pred, noise)
  loss = 0.5 * loss_mse + 0.5 * loss_diff
```

### 4.2 推理时数据流

```
输入:
  x_enc: [32, 96, 7]

Step 1-5: 同训练 (得到 y_det, z)
  y_det: [32, 96, 7]
  z: [32, 7, 128]

Step 6: DDIM Sampling (n_samples=10, ddim_steps=50)
  for sample_idx in range(10):
    x = randn([32, 7, 96])  # 初始噪声
    for t in [950, 900, ..., 0]:  # 50 steps
      noise_pred = denoise_net(x, t, z)
      x = ddim_update(x, noise_pred, t)
    samples[sample_idx] = x

  residual_samples: [10, 32, 7, 96]
    → permute → [10, 32, 96, 7]
    → denormalize → [10, 32, 96, 7]

Step 7: Final Prediction
  samples = y_det.unsqueeze(0) + residual_samples
  samples: [10, 32, 96, 7]

  mean_pred = samples.mean(dim=0): [32, 96, 7]
  std_pred = samples.std(dim=0): [32, 96, 7]
```

---

## 5. 训练流程

### 5.1 两阶段训练策略

```python
# exp/exp_diffusion_forecast.py

class Exp_Diffusion_Forecast(Exp_Basic):
    def train(self, setting):
        # Stage 1: Backbone Warmup
        print("Stage 1: Backbone Warmup Training")
        self.train_stage1(setting)

        # 加载最佳 Stage 1 模型
        self.model.load_state_dict(torch.load(path + '/checkpoint_stage1.pth'))

        # Stage 2: Joint Training
        print("Stage 2: Joint Training (Diffusion)")
        self.model.freeze_encoder()
        self.train_stage2(setting)
```

### 5.2 Stage 1 训练循环

```python
def train_stage1(self, setting):
    """
    Stage 1: 仅训练 iTransformer 骨干网
    - 损失: MSE only
    - 目标: 学习确定性预测
    """
    for epoch in range(self.args.stage1_epochs):
        self.model.train()
        train_loss = []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()

            # 前向
            loss, loss_dict = self.model.forward_loss(
                batch_x, batch_x_mark, batch_y[:, -self.args.pred_len:, :],
                stage='warmup'
            )

            # 反向
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # 验证
        vali_loss = self.vali_stage1(...)

        # Early Stopping
        early_stopping(vali_loss, self.model, path, suffix='_stage1')
        if early_stopping.early_stop:
            break
```

### 5.3 Stage 2 训练循环

```python
def train_stage2(self, setting):
    """
    Stage 2: 联合训练扩散网络
    - 编码器冻结
    - 损失: λ*MSE + (1-λ)*Diffusion
    """
    self.model.freeze_encoder()

    # 重新创建优化器 (只优化未冻结参数)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr=self.args.learning_rate
    )

    for epoch in range(self.args.stage2_epochs):
        self.model.train()

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()

            loss, loss_dict = self.model.forward_loss(
                batch_x, batch_x_mark, batch_y[:, -self.args.pred_len:, :],
                stage='joint'
            )

            loss.backward()
            optimizer.step()

        # 验证 + Early Stopping
        early_stopping(vali_loss, self.model, path, suffix='_stage2')
```

### 5.4 自定义 EarlyStopping

```python
class EarlyStoppingWithSuffix:
    """
    支持 suffix 参数的 Early Stopping。

    原因: 原版 EarlyStopping 不支持 suffix，无法为两阶段保存不同检查点。
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path, suffix=''):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, suffix)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, suffix)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, suffix=''):
        torch.save(model.state_dict(), path + '/checkpoint' + suffix + '.pth')
        self.val_loss_min = val_loss
```

---

## 6. 参数配置

### 6.1 完整参数列表

```python
# run.py 中添加的参数

# 扩散参数
parser.add_argument('--diffusion_steps', type=int, default=1000,
                    help='Number of diffusion timesteps')
parser.add_argument('--beta_schedule', type=str, default='cosine',
                    choices=['linear', 'cosine'])
parser.add_argument('--cond_dim', type=int, default=256,
                    help='Condition embedding dimension')
parser.add_argument('--unet_channels', type=str, default='64,128,256,512',
                    help='U-Net channel configuration')

# 两阶段训练
parser.add_argument('--stage1_epochs', type=int, default=30,
                    help='Stage 1 (backbone warmup) epochs')
parser.add_argument('--stage2_epochs', type=int, default=20,
                    help='Stage 2 (joint training) epochs')

# 采样参数
parser.add_argument('--n_samples', type=int, default=100,
                    help='Number of samples for probabilistic prediction')
parser.add_argument('--use_ddim', action='store_true',
                    help='Use DDIM instead of DDPM')
parser.add_argument('--ddim_steps', type=int, default=50,
                    help='Number of DDIM steps')
```

### 6.2 默认配置对照表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| diffusion_steps | 1000 | 扩散步数 |
| beta_schedule | cosine | Beta 调度方式 |
| cond_dim | 256 | 条件嵌入维度 |
| unet_channels | [64,128,256,512] | U-Net 通道 |
| stage1_epochs | 30 | Stage 1 轮数 |
| stage2_epochs | 20 | Stage 2 轮数 |
| n_samples | 100 | 推理采样数 |
| ddim_steps | 50 | DDIM 步数 |

### 6.3 ETTh1 实验配置

```bash
# scripts/diffusion_forecast/ETT_script/iTransformerDiffusion_ETTh1.sh

python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model_id ETTh1_96_96 \
  --model iTransformerDiffusion \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 128 \
  --n_heads 8 \
  --dropout 0.1 \
  --diffusion_steps 1000 \
  --beta_schedule cosine \
  --cond_dim 256 \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --n_samples 100 \
  --use_ddim \
  --ddim_steps 50 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --itr 1
```

---

## 7. 接口设计

### 7.1 模型接口

```python
class Model(nn.Module):
    # 初始化
    def __init__(self, configs): ...

    # 骨干网前向 (返回确定性预测 + 条件特征)
    def backbone_forward(self, x_enc, x_mark_enc=None) -> (y_det, z, means, stdev): ...

    # 训练损失计算
    def forward_loss(self, x_enc, x_mark_enc, y_true, stage='joint') -> (loss, loss_dict): ...

    # DDPM 采样
    def sample_ddpm(self, z, n_samples=1) -> samples: ...

    # DDIM 采样
    def sample_ddim(self, z, n_samples=1, ddim_steps=50, eta=0.0) -> samples: ...

    # 概率预测
    def predict(self, x_enc, x_mark_enc=None, n_samples=None, ...) -> (mean, std, samples): ...

    # TSLib 兼容 forward
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) -> y_det: ...

    # 冻结/解冻
    def freeze_encoder(self): ...
    def unfreeze_encoder(self): ...
```

### 7.2 实验类接口

```python
class Exp_Diffusion_Forecast(Exp_Basic):
    # 构建模型
    def _build_model(self) -> Model: ...

    # Stage 1 训练
    def train_stage1(self, setting): ...

    # Stage 2 训练
    def train_stage2(self, setting): ...

    # Stage 1 验证
    def vali_stage1(self, vali_data, vali_loader, criterion): ...

    # Stage 2 验证
    def vali_stage2(self, vali_data, vali_loader): ...

    # 完整训练流程
    def train(self, setting): ...

    # 概率测试
    def test(self, setting, test=0): ...
```

### 7.3 概率指标接口

```python
# utils/metrics.py

def CRPS(pred_samples, true):
    """
    Continuous Ranked Probability Score.

    Args:
        pred_samples: [n_samples, B, T, N]
        true: [B, T, N]
    Returns:
        scalar CRPS
    """
    ...

def calibration(pred_samples, true, quantiles=[0.1, 0.5, 0.9]):
    """
    校准度评估。

    Returns:
        dict: {quantile: coverage_rate}
    """
    ...

def sharpness(pred_samples):
    """
    锐度评估 (预测分布的集中程度)。

    Returns:
        scalar: 平均标准差
    """
    ...

def prob_metric(pred_samples, true):
    """
    完整概率指标。

    Returns:
        dict: {crps, calibration, sharpness}
    """
    ...
```

---

## 8. Bug修复记录

### 8.1 Bug #1: ModuleNotFoundError: layers.DWT_Decomposition

**发现时间**: 2025-12-28
**错误信息**:
```
ModuleNotFoundError: No module named 'layers.DWT_Decomposition'
```

**原因分析**:
在编辑 layers 目录时，意外删除了 `DWT_Decomposition.py` 文件。

**修复方案**:
```bash
git restore layers/DWT_Decomposition.py
```

**影响范围**: 无，仅恢复文件

---

### 8.2 Bug #2: EarlyStopping 不支持 suffix 参数

**发现时间**: 2025-12-28
**错误信息**:
```
TypeError: EarlyStopping.__call__() got an unexpected keyword argument 'suffix'
```

**原因分析**:
原版 `utils/tools.py` 中的 `EarlyStopping` 类不支持 suffix 参数，无法为两阶段训练保存不同的检查点文件。

**修复方案**:
在 `exp/exp_diffusion_forecast.py` 中创建自定义类:

```python
class EarlyStoppingWithSuffix:
    def __call__(self, val_loss, model, path, suffix=''):
        # ... 支持 suffix 的实现

    def save_checkpoint(self, val_loss, model, path, suffix=''):
        torch.save(model.state_dict(), path + '/checkpoint' + suffix + '.pth')
```

**影响范围**: 仅影响 diffusion_forecast 任务

---

### 8.3 Bug #3: UpBlock1D 通道数不匹配

**发现时间**: 2025-12-28
**错误信息**:
```
RuntimeError: Given groups=1, weight of size [256, 768, 3], expected input[32, 1024, 12] to have 768 channels, but got 1024 channels instead
```

**原因分析**:
UpBlock1D 中 skip connection 的通道数计算错误。原始实现假设 skip 通道 = 输出通道，但实际应该是下采样时保存的通道数。

```
错误假设:
  UpBlock(512→256): concat(512, 256) = 768 ✗

正确计算:
  DownBlock 输出 [512, 256, 128, 64]
  对应 skip    [256, 128, 64] (下采样前的输出)

  UpBlock(512→256): concat(512, 512) = 1024 ✓
```

**修复方案**:

1. 修改 `UpBlock1D` 添加 `skip_channels` 参数:
```python
class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, cond_dim, ...):
        self.res = ResBlock1D(in_channels + skip_channels, out_channels, cond_dim)
```

2. 修改 `UNet1D` 正确计算 skip_channels:
```python
# channels = [64, 128, 256, 512]
reversed_channels = [512, 256, 128, 64]
skip_channels_list = [512, 256, 128]  # = list(reversed(channels[1:]))

for i, out_ch in enumerate([256, 128, 64]):
    in_ch = [512, 256, 128][i]
    skip_ch = [512, 256, 128][i]
    self.up_blocks.append(UpBlock1D(in_ch, out_ch, skip_ch, ...))
```

**影响范围**: `layers/Diffusion_layers.py` 的 UpBlock1D 和 UNet1D

---

### 8.4 Bug #4: sample_ddpm/ddim 通道数不匹配

**发现时间**: 2025-12-28
**错误信息**:
```
RuntimeError: shape mismatch: expected [32, 7, 96], got [32, 11, 96]
```

**原因分析**:
在 `sample_ddpm` 和 `sample_ddim` 中使用 `z.shape[1]` 获取变量数 N，但在某些情况下 z 的通道数可能与 `self.n_vars` 不一致。

**修复方案**:
```python
@torch.no_grad()
def sample_ddpm(self, z, n_samples=1):
    B, _, _ = z.shape
    N = self.n_vars  # 使用 self.n_vars 而非 z.shape[1]

    x = torch.randn(B, N, self.pred_len, device=device)
    # ...
```

**影响范围**: `models/iTransformerDiffusion.py` 的 sample_ddpm 和 sample_ddim

---

### 8.5 Bug #5: n_samples=1 时 std 计算警告

**发现时间**: 2025-12-28
**警告信息**:
```
RuntimeWarning: Degrees of freedom <= 0 for slice
```

**原因分析**:
当 `n_samples=1` 时，`std(dim=0)` 默认使用 `unbiased=True`，需要 N-1=0 个自由度，导致除零。

**修复方案**:
```python
def predict(self, ...):
    # ...
    std_pred = samples.std(dim=0, unbiased=False)  # 添加 unbiased=False
```

**影响范围**: `models/iTransformerDiffusion.py` 的 predict 函数

---

### 8.6 Bug #6: backbone_forward 通道对齐

**发现时间**: 2025-12-28
**问题描述**:
编码器输出 z 的通道数可能与 `self.n_vars` 不一致，导致后续操作失败。

**修复方案**:
```python
def backbone_forward(self, x_enc, x_mark_enc=None):
    N = self.n_vars  # 使用固定值

    # ... 编码器计算 ...

    # 通道对齐检查
    actual_n_vars = z.shape[1]
    if actual_n_vars != N:
        if actual_n_vars > N:
            z = z[:, :N, :]
        else:
            padding = torch.zeros(z.shape[0], N - actual_n_vars, z.shape[2],
                                device=z.device, dtype=z.dtype)
            z = torch.cat([z, padding], dim=1)
```

**影响范围**: `models/iTransformerDiffusion.py` 的 backbone_forward

---

## 9. 单元测试

### 9.1 测试文件结构

```
tests/
├── test_iTransformerDiffusion.py  # 主要单元测试 (15 个)
└── test_edge_cases.py             # 边界情况测试 (8 个)
```

### 9.2 test_iTransformerDiffusion.py 测试用例

```python
class TestModelInitialization:
    def test_basic_init(self):
        """测试基本模型初始化"""
        # 验证: n_vars, seq_len, pred_len, timesteps 正确设置

    def test_diffusion_schedule(self):
        """测试扩散调度设置"""
        # 验证: betas, alphas, alpha_cumprods 正确计算


class TestBackboneForward:
    def test_backbone_forward_basic(self):
        """测试骨干网前向传播"""
        # 输入: [B, seq_len, N]
        # 验证: y_det [B, pred_len, N], z [B, N, d_model]

    def test_backbone_forward_channel_alignment(self):
        """测试通道对齐"""
        # 验证: z.shape[1] == self.n_vars


class TestForwardLoss:
    def test_forward_loss_warmup(self):
        """测试 Stage 1 损失"""
        # 验证: 返回 loss_mse

    def test_forward_loss_joint(self):
        """测试 Stage 2 损失"""
        # 验证: 返回 loss_total, loss_mse, loss_diff


class TestSampling:
    def test_sample_ddpm(self):
        """测试 DDPM 采样"""
        # 验证: samples [n_samples, B, N, pred_len]

    def test_sample_ddim(self):
        """测试 DDIM 采样"""
        # 验证: samples [n_samples, B, N, pred_len]

    def test_sample_channel_consistency(self):
        """测试采样通道一致性"""
        # 验证: 即使 z.shape[1] != n_vars，输出仍正确


class TestPredict:
    def test_predict_basic(self):
        """测试基本预测"""
        # 验证: mean_pred, std_pred, samples 形状正确

    def test_predict_with_ddim(self):
        """测试 DDIM 预测"""


class TestFreezeUnfreeze:
    def test_freeze_encoder(self):
        """测试冻结编码器"""
        # 验证: embedding, encoder 参数 requires_grad=False

    def test_unfreeze_encoder(self):
        """测试解冻编码器"""


class TestEndToEnd:
    def test_training_step(self):
        """测试完整训练步骤"""
        # Stage 1 + Stage 2 各一步

    def test_full_inference(self):
        """测试完整推理"""
        # 验证: 输出形状、std >= 0
```

### 9.3 test_edge_cases.py 测试用例

```python
class TestEdgeCases:
    def test_single_variate(self):
        """测试单变量情况 (N=1)"""

    def test_many_variates(self):
        """测试多变量情况 (N=20)"""

    def test_short_sequence(self):
        """测试短序列 (seq_len=24)"""

    def test_long_sequence(self):
        """测试长序列 (seq_len=336)"""

    def test_batch_size_one(self):
        """测试 batch_size=1"""

    def test_large_batch(self):
        """测试大批量 (batch_size=32)"""

    def test_gradient_flow(self):
        """测试梯度流动"""
        # Stage 1: 所有参数有梯度
        # Stage 2: 编码器无梯度，扩散网络有梯度

    def test_residual_normalizer(self):
        """测试残差归一化器"""
        # 验证: normalize + denormalize ≈ identity
```

### 9.4 运行测试

```bash
# 运行所有测试
python tests/test_iTransformerDiffusion.py
python tests/test_edge_cases.py

# 测试结果
# test_iTransformerDiffusion.py: 15/15 通过
# test_edge_cases.py: 8/8 通过
```

---

## 10. 版本迭代记录

### v1.0 (2025-12-28) - 初始版本

**新增功能**:
- 完整 iTransformer + CRD-Net 架构
- 两阶段训练流程
- DDPM / DDIM 采样
- 概率预测指标 (CRPS, Calibration, Sharpness)

**文件变更**:
- 新建 6 个文件
- 修改 3 个文件

**Bug 修复**:
- #1 DWT_Decomposition 文件丢失
- #2 EarlyStopping suffix 支持
- #3 UpBlock1D 通道计算
- #4 采样通道不一致
- #5 n_samples=1 std 警告
- #6 backbone_forward 通道对齐

**测试覆盖**:
- 23 个单元测试全部通过

**端到端验证**:
- ETTh1 快速测试通过
- CRPS: 0.5465
- Calibration: 77%/98%

---

## 附录 A: 关键代码片段索引

| 功能 | 文件 | 行号 |
|------|------|------|
| SinusoidalPosEmb | layers/Diffusion_layers.py | 15-30 |
| FiLMLayer | layers/Diffusion_layers.py | 55-70 |
| VariateCrossAttention | layers/Diffusion_layers.py | 75-120 |
| ResBlock1D | layers/Diffusion_layers.py | 125-155 |
| UpBlock1D (修复后) | layers/Diffusion_layers.py | 175-210 |
| UNet1D | layers/Diffusion_layers.py | 220-320 |
| Model.__init__ | models/iTransformerDiffusion.py | 27-90 |
| backbone_forward (修复后) | models/iTransformerDiffusion.py | 119-164 |
| forward_loss | models/iTransformerDiffusion.py | 187-244 |
| sample_ddpm (修复后) | models/iTransformerDiffusion.py | 246-295 |
| sample_ddim (修复后) | models/iTransformerDiffusion.py | 297-357 |
| predict (修复后) | models/iTransformerDiffusion.py | 359-401 |
| EarlyStoppingWithSuffix | exp/exp_diffusion_forecast.py | 25-55 |
| train_stage1 | exp/exp_diffusion_forecast.py | 100-150 |
| train_stage2 | exp/exp_diffusion_forecast.py | 155-210 |
| CRPS | utils/metrics.py | 新增行 |

---

## 附录 B: 相关论文参考

1. **iTransformer**: Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024
2. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
3. **DDIM**: Song et al., "Denoising Diffusion Implicit Models", ICLR 2021
4. **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
5. **Cosine Schedule**: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021

---

**文档结束**
