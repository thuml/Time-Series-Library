# iTransformerDiffusionDirect 技术文档

> **模型定位**: 基于 iTransformer 骨干网络的直接预测扩散模型 (x₀-parameterization)
> **适用任务**: 概率时序预测 (Probabilistic Time Series Forecasting)
> **作者**: Time-Series-Library 研究团队
> **版本**: v1.0

---

## 目录

1. [模型概述](#1-模型概述)
2. [核心架构](#2-核心架构)
3. [与 ε-prediction 版本的对比](#3-与-ε-prediction-版本的对比)
4. [数学原理](#4-数学原理)
5. [代码实现详解](#5-代码实现详解)
6. [训练策略](#6-训练策略)
7. [实验结果分析](#7-实验结果分析)
8. [已知问题与改进方向](#8-已知问题与改进方向)
9. [使用指南](#9-使用指南)
10. [参考文献](#10-参考文献)

---

## 1. 模型概述

### 1.1 设计动机

iTransformerDiffusionDirect 是一种结合 **iTransformer** 时序编码能力与 **扩散模型** 概率生成能力的混合架构。与传统的 ε-prediction 扩散模型不同，本模型采用 **x₀-parameterization**，即直接预测干净数据而非噪声。

### 1.2 核心特点

| 特点 | 描述 |
|------|------|
| **x₀-parameterization** | 网络直接预测干净数据 x₀，而非噪声 ε |
| **iTransformer 骨干** | 利用变量级注意力机制提取条件特征 |
| **两阶段训练** | Stage 1 预热骨干网络，Stage 2 联合训练扩散 |
| **概率预测** | 通过多次采样获得预测分布，量化不确定性 |

### 1.3 模型定位

```
确定性预测                    概率预测
    │                           │
    ▼                           ▼
iTransformer ──────────> iTransformerDiffusionDirect
  (点估计)                    (分布估计)
```

---

## 2. 核心架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    iTransformerDiffusionDirect               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: x_hist [B, seq_len, N]                              │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────────────────┐                   │
│  │       iTransformer Backbone           │                   │
│  │  ┌────────────────────────────────┐  │                   │
│  │  │  Instance Normalization         │  │                   │
│  │  │  DataEmbedding_inverted         │  │                   │
│  │  │  Encoder (attention on variates)│  │                   │
│  │  │  Projection                     │  │                   │
│  │  └────────────────────────────────┘  │                   │
│  └──────────────────────────────────────┘                   │
│      │                    │                                  │
│      ▼                    ▼                                  │
│  y_det [B, pred_len, N]   z [B, N, d_model]                 │
│  (仅 Stage 1 使用)        (条件特征)                         │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────┐                   │
│  │     Direct Prediction Diffusion       │                   │
│  │  ┌────────────────────────────────┐  │                   │
│  │  │  Noise: ε ~ N(0, I)            │  │                   │
│  │  │  Forward: xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε │  │                │
│  │  │  UNet1D: x₀_pred = f(xₜ, t, z) │  │                   │
│  │  │  Reverse: DDPM/DDIM sampling   │  │                   │
│  │  └────────────────────────────────┘  │                   │
│  └──────────────────────────────────────┘                   │
│      │                                                       │
│      ▼                                                       │
│  samples [n_samples, B, pred_len, N]                        │
│  → mean_pred, std_pred                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 组件说明

#### 2.2.1 iTransformer Backbone

| 组件 | 类名 | 作用 |
|------|------|------|
| 嵌入层 | `DataEmbedding_inverted` | 将时间维度投影到 d_model |
| 编码器 | `Encoder` | 在变量维度上进行自注意力 |
| 投影层 | `nn.Linear` | 输出确定性预测 (Stage 1) |

**数据流**:
```python
x_enc [B, seq_len, N]
  → Instance Norm
  → permute → [B, N, seq_len]
  → Linear(seq_len → d_model) → [B, N, d_model]
  → Encoder (attention across N variates)
  → z [B, N, d_model]  # 条件特征
  → Projection → y_det [B, pred_len, N]
```

#### 2.2.2 UNet1D 去噪网络

| 组件 | 作用 |
|------|------|
| `SinusoidalPosEmb` | 时间步 t 的正弦位置编码 |
| `ConditionProjector` | 融合 z 和 t_emb → 全局条件向量 |
| `DownBlock1D` | 下采样 + ResBlock + FiLM |
| `UpBlock1D` | 上采样 + ResBlock + FiLM + CrossAttn |
| `FiLMLayer` | γ·h + β 特征调制 |
| `VariateCrossAttention` | 变量级交叉注意力 |

**UNet1D 结构**:
```
Input: x_noisy [B, N, pred_len]
       t [B]
       z [B, N, d_model]
           │
           ▼
    ┌──────────────┐
    │  init_conv   │  N → 64
    └──────────────┘
           │
    ┌──────┴──────┐
    │  DownBlock  │  64 → 128 → 256 → 512
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ Bottleneck  │  ResBlock + CrossAttn
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │   UpBlock   │  512 → 256 → 128 → 64
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ final_conv  │  64 → N
    └─────────────┘
           │
           ▼
Output: x₀_pred [B, N, pred_len]
```

---

## 3. 与 ε-prediction 版本的对比

### 3.1 核心区别

| 方面 | iTransformerDiffusion (ε) | iTransformerDiffusionDirect (x₀) |
|------|---------------------------|----------------------------------|
| **预测目标** | 噪声 ε | 干净数据 x₀ |
| **训练损失** | MSE(ε_pred, ε) | MSE(x₀_pred, x₀) |
| **采样过程** | 从 ε_pred 推导 x₀ | 直接使用 x₀_pred |
| **数值稳定性** | 较好 | 需要额外 clipping |
| **采样质量** | 理论上更好 | 可能略有下降 |

### 3.2 训练损失对比

**ε-prediction (iTransformerDiffusion)**:
```python
# 训练时
noise = torch.randn_like(x0)
xt = sqrt(ᾱt) * x0 + sqrt(1-ᾱt) * noise
noise_pred = model(xt, t, z)
loss = MSE(noise_pred, noise)
```

**x₀-prediction (iTransformerDiffusionDirect)**:
```python
# 训练时
noise = torch.randn_like(x0)
xt = sqrt(ᾱt) * x0 + sqrt(1-ᾱt) * noise
x0_pred = model(xt, t, z)
loss = MSE(x0_pred, x0)  # 直接预测干净数据
```

### 3.3 采样过程对比

**ε-prediction 采样**:
```python
# 每个时间步
noise_pred = model(xt, t, z)
# 需要从 noise_pred 推导 mean
mean = (1/√αt) * (xt - βt/√(1-ᾱt) * noise_pred)
```

**x₀-prediction 采样**:
```python
# 每个时间步
x0_pred = model(xt, t, z)
x0_pred = clamp(x0_pred, -3, 3)  # 稳定性
# 从 x0_pred 推导 noise_pred
noise_pred = (xt - √ᾱt * x0_pred) / √(1-ᾱt)
# 然后正常 DDPM/DDIM 更新
```

### 3.4 优劣势分析

| 方面 | ε-prediction | x₀-prediction |
|------|:------------:|:-------------:|
| 训练稳定性 | ✅ 更好 | ⚠️ 需要 clipping |
| 采样质量 | ✅ 理论最优 | ⚠️ 可能略低 |
| 实现复杂度 | ⚠️ 更复杂 | ✅ 更直接 |
| 可解释性 | ⚠️ 预测噪声 | ✅ 直接预测目标 |
| 条件生成 | ⚠️ 间接 | ✅ 直接约束输出 |

---

## 4. 数学原理

### 4.1 前向扩散过程

给定干净数据 x₀，前向扩散在 T 步内逐渐添加高斯噪声：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$$

其中：
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

**重参数化采样**:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### 4.2 逆向去噪过程

**x₀-parameterization** 的核心是训练网络直接预测 x₀：

$$\hat{x}_0 = f_\theta(x_t, t, z)$$

其中 z 是 iTransformer 编码器的条件特征。

**从 x₀ 推导噪声估计**:
$$\hat{\epsilon} = \frac{x_t - \sqrt{\bar{\alpha}_t} \hat{x}_0}{\sqrt{1-\bar{\alpha}_t}}$$

**DDPM 更新**:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon} \right) + \sigma_t z$$

### 4.3 训练目标

**Stage 1 (Backbone Warmup)**:
$$\mathcal{L}_1 = \mathbb{E}_{x, y} \left[ \| y_{det} - y_{true} \|^2 \right]$$

**Stage 2 (Joint Training)**:
$$\mathcal{L}_2 = \lambda \cdot \mathcal{L}_{MSE} + (1-\lambda) \cdot \mathcal{L}_{diff}$$

其中：
$$\mathcal{L}_{diff} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| f_\theta(x_t, t, z) - x_0 \|^2 \right]$$

---

## 5. 代码实现详解

### 5.1 模型初始化

```python
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # 基础配置
        self.seq_len = configs.seq_len      # 输入序列长度
        self.pred_len = configs.pred_len    # 预测序列长度
        self.n_vars = configs.enc_in        # 变量数量
        self.d_model = configs.d_model      # 模型维度

        # 扩散配置
        self.timesteps = getattr(configs, 'diffusion_steps', 1000)
        self.beta_schedule = getattr(configs, 'beta_schedule', 'cosine')

        # iTransformer 骨干
        self.enc_embedding = DataEmbedding_inverted(...)
        self.encoder = Encoder(...)
        self.projection = nn.Linear(d_model, pred_len)

        # UNet1D 去噪网络
        self.denoise_net = UNet1D(
            n_vars=self.n_vars,
            pred_len=self.pred_len,
            d_model=self.d_model,
            cond_dim=256,
            channels=[64, 128, 256, 512]
        )

        # 扩散调度
        self._setup_diffusion_schedule()
```

### 5.2 骨干网络前向传播

```python
def backbone_forward(self, x_enc, x_mark_enc=None):
    """
    Args:
        x_enc: [B, seq_len, N] 输入历史
        x_mark_enc: [B, seq_len, M] 时间标记
    Returns:
        y_det: [B, pred_len, N] 确定性预测
        z: [B, N, d_model] 编码器特征 (条件)
        means, stdev: 归一化统计量
    """
    # 实例归一化
    means = x_enc.mean(1, keepdim=True).detach()
    x_enc = x_enc - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True) + 1e-5)
    x_enc = x_enc / stdev

    # 嵌入: [B, seq_len, N] -> [B, N, d_model]
    enc_out = self.enc_embedding(x_enc, x_mark_enc)

    # 编码器: [B, N, d_model] -> [B, N, d_model]
    z, _ = self.encoder(enc_out)

    # 投影: [B, N, d_model] -> [B, pred_len, N]
    y_det = self.projection(z).permute(0, 2, 1)

    # 反归一化
    y_det = y_det * stdev + means

    return y_det, z, means, stdev
```

### 5.3 训练损失计算

```python
def forward_loss(self, x_enc, x_mark_enc, y_true, stage='joint'):
    B = x_enc.shape[0]
    device = x_enc.device

    # 骨干网络前向
    y_det, z, means, stdev = self.backbone_forward(x_enc, x_mark_enc)

    # MSE 损失
    loss_mse = F.mse_loss(y_det, y_true)

    if stage == 'warmup':
        return loss_mse, {'loss_mse': loss_mse.item()}

    # Stage 2: x₀ 预测扩散
    # 归一化目标
    y_norm = (y_true - means) / stdev
    y_norm = y_norm.permute(0, 2, 1)  # [B, N, pred_len]

    # 随机时间步
    t = torch.randint(0, self.timesteps, (B,), device=device)

    # 加噪
    noise = torch.randn_like(y_norm)
    y_noisy, _ = self.add_noise(y_norm, t, noise)

    # 预测 x₀ (不是噪声!)
    x0_pred = self.denoise_net(y_noisy, t, z)

    # 扩散损失
    loss_diff = F.mse_loss(x0_pred, y_norm)

    # 组合损失
    loss_total = 0.5 * loss_mse + 0.5 * loss_diff

    return loss_total, {...}
```

### 5.4 DDPM 采样

```python
@torch.no_grad()
def sample_ddpm_x0(self, z, n_samples=1):
    B, _, _ = z.shape
    device = z.device
    N = self.n_vars

    all_samples = []
    for _ in range(n_samples):
        # 从纯噪声开始
        x = torch.randn(B, N, self.pred_len, device=device)

        # 逆向扩散
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((B,), t, device=device)

            # 直接预测 x₀
            x0_pred = self.denoise_net(x, t_batch, z)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)  # 稳定性

            # 从 x₀ 推导噪声
            alpha_t = self.alpha_cumprods[t]
            noise_pred = (x - sqrt(alpha_t) * x0_pred) / sqrt(1 - alpha_t)

            # DDPM 更新
            mean = (1/sqrt(αt)) * (x - βt/sqrt(1-ᾱt) * noise_pred)

            if t > 0:
                x = mean + sqrt(βt) * torch.randn_like(x)
            else:
                x = mean

        all_samples.append(x)

    return torch.stack(all_samples, dim=0)
```

---

## 6. 训练策略

### 6.1 两阶段训练

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Stage 1: Backbone Warmup (30 epochs)                   │
│  ├── 训练参数: enc_embedding + encoder + projection     │
│  ├── 损失函数: MSE(y_det, y_true)                       │
│  ├── 学习率: 1e-4 (AdamW)                               │
│  └── 目的: 让 iTransformer 学会基础预测                  │
│                                                          │
│  Stage 2: Joint Diffusion (20 epochs)                   │
│  ├── 冻结参数: enc_embedding + encoder                  │
│  ├── 训练参数: projection + denoise_net                 │
│  ├── 损失函数: 0.5*MSE + 0.5*Diffusion                  │
│  ├── 学习率: projection=1e-5, diffusion=1e-4            │
│  └── 目的: 训练扩散模型学习条件生成                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 6.2 优化器配置

```python
# Stage 1
optimizer_stage1 = AdamW(
    [enc_embedding, encoder, projection],
    lr=1e-4,
    weight_decay=0.01
)
scheduler_stage1 = CosineAnnealingLR(T_max=30)

# Stage 2
optimizer_stage2 = AdamW([
    {'params': projection.parameters(), 'lr': 1e-5},
    {'params': denoise_net.parameters(), 'lr': 1e-4},
], weight_decay=0.01)
scheduler_stage2 = CosineAnnealingLR(T_max=20)
```

### 6.3 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `diffusion_steps` | 1000 | 扩散步数 |
| `beta_schedule` | cosine | β 调度方式 |
| `d_model` | 128 | 模型维度 |
| `cond_dim` | 256 | 条件维度 |
| `unet_channels` | [64,128,256,512] | UNet 通道数 |
| `stage1_epochs` | 30 | Stage 1 轮数 |
| `stage2_epochs` | 20 | Stage 2 轮数 |
| `loss_lambda` | 0.5 | MSE 损失权重 |
| `n_samples` | 100 | 采样数量 |

---

## 7. 实验结果分析

### 7.1 当前实验配置

```
数据集: ETTh1 (7 variates)
任务: 96 → 96 长期预测
模型: iTransformerDiffusionDirect
配置: d_model=128, e_layers=2, diffusion_steps=1000
```

### 7.2 点预测性能

| 模型 | MSE | MAE | RMSE |
|------|----:|----:|-----:|
| **iTransformerDiffusionDirect** | 0.5995 | 0.5017 | 0.7743 |
| iTransformerDiffusion (AMP) | 0.4270 | 0.4374 | 0.6534 |
| iTransformer (baseline) | 0.3853 | 0.4033 | 0.6207 |
| WPMixer (baseline) | 0.3749 | 0.4029 | 0.6123 |

**分析**:
- Direct 版本 MSE=0.5995，比 ε-prediction 版本高 40%
- 比基线 iTransformer 高 55.6%
- 点预测性能较差，但符合预期 (扩散模型优势在概率预测)

### 7.3 概率预测性能

| 指标 | iTransformerDiffusionDirect | 理想值 |
|------|----------------------------:|-------:|
| CRPS | 0.4949 | 越低越好 |
| Calibration 50% | 0.4901 | 0.5000 |
| Calibration 90% | 0.8787 | 0.9000 |
| Sharpness | 0.6305 | 越低越好 |

**分析**:
- Calibration 50% 偏差仅 1.0%，非常接近理想值
- Calibration 90% 偏差 2.1%，较好
- CRPS=0.495 与 ε-prediction 版本相当

### 7.4 按变量 MSE 分析

| 变量 | MSE | 说明 |
|------|----:|------|
| 变量 0 | 1.3539 | 较难预测 |
| 变量 1 | 0.3033 | 中等 |
| 变量 2 | 1.4440 | 最难预测 |
| 变量 3 | 0.2468 | 较好 |
| 变量 4 | 0.6461 | 中等 |
| 变量 5 | 0.1417 | 较好 |
| 变量 6 | 0.0610 | 最好 |

**发现**: 变量 0 和 2 的 MSE 显著高于其他变量，可能需要针对性优化。

### 7.5 预测分布分析

```
预测值统计:
  范围: [-3.42, 3.05]
  均值: 0.0498
  标准差: 0.7279

真实值统计:
  范围: [-4.68, 4.07]
  均值: 0.0395
  标准差: 1.0528

问题: 预测分布的标准差 (0.73) 小于真实值 (1.05)
原因: 模型倾向于预测更保守的值，欠拟合极端情况
```

---

## 8. 已知问题与改进方向

### 8.1 当前问题

| 问题 | 严重程度 | 描述 |
|------|:--------:|------|
| 点预测性能差 | ⚠️ | MSE 比基线高 55%+ |
| 预测范围窄 | ⚠️ | 预测 std=0.73 < 真实 std=1.05 |
| 变量不平衡 | ⚠️ | 变量 0,2 MSE 显著高于其他 |
| 训练成本高 | ⚡ | 两阶段训练 + 1000 步扩散 |

### 8.2 改进方向

#### 短期改进 (Quick Wins)

| 建议 | 预期效果 | 实现难度 |
|------|----------|:--------:|
| 增加 Stage 2 轮数 (20→40) | 提高扩散质量 | ⭐ |
| 使用 DDIM 加速采样 (50步) | 降低推理时间 | ⭐ |
| 调整 clipping 范围 (-3,3→-4,4) | 扩大预测范围 | ⭐ |
| 降低扩散学习率 (1e-4→5e-5) | 提高稳定性 | ⭐ |

#### 中期改进 (Architecture)

| 建议 | 描述 | 实现难度 |
|------|------|:--------:|
| 变量加权损失 | 对难预测变量给予更高权重 | ⭐⭐ |
| 混合预测 (ε + x₀) | v-prediction parameterization | ⭐⭐⭐ |
| 自适应 β schedule | 根据数据特征调整噪声调度 | ⭐⭐ |
| 残差连接增强 | 添加 backbone 预测到最终输出 | ⭐⭐ |

#### 长期改进 (Research)

| 方向 | 潜力 | 说明 |
|------|:----:|------|
| Flow Matching | ⭐⭐⭐ | ODE-based，比 DDPM 更快更稳 |
| Consistency Models | ⭐⭐⭐ | 单步生成，大幅加速 |
| Score Distillation | ⭐⭐ | 蒸馏到小模型 |
| Diffusion Transformer | ⭐⭐ | 用 Transformer 替代 UNet |

### 8.3 推荐优先级

```
1. [P0] 增加 Stage 2 训练轮数
2. [P0] 实现 DDIM 加速采样
3. [P1] 变量加权损失
4. [P1] 调整 clipping 和 β schedule
5. [P2] 研究 v-prediction
6. [P3] 探索 Flow Matching
```

---

## 9. 使用指南

### 9.1 训练命令

```bash
# 激活环境
conda activate tslib

# 完整训练
python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model iTransformerDiffusionDirect \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 96 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 128 --d_ff 128 \
  --e_layers 2 --n_heads 8 \
  --diffusion_steps 1000 \
  --beta_schedule cosine \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --n_samples 100 \
  --batch_size 32

# 低显存版本 (添加 AMP)
python run.py ... --use_amp

# 仅测试
python run.py --is_training 0 ...
```

### 9.2 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--diffusion_steps` | 扩散步数 | 1000 |
| `--beta_schedule` | β 调度 | cosine |
| `--stage1_epochs` | Stage 1 轮数 | 30 |
| `--stage2_epochs` | Stage 2 轮数 | 20-40 |
| `--n_samples` | 采样数量 | 100 |
| `--use_ddim` | 使用 DDIM | 推理时启用 |
| `--ddim_steps` | DDIM 步数 | 50 |
| `--use_amp` | 混合精度 | 低显存时启用 |

### 9.3 代码示例

```python
# 加载模型
from models.iTransformerDiffusionDirect import Model

model = Model(configs)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# 概率预测
with torch.no_grad():
    mean_pred, std_pred, samples = model.predict(
        x_enc,
        x_mark_enc,
        n_samples=100,
        use_ddim=True,
        ddim_steps=50
    )

# mean_pred: [B, pred_len, N] 均值预测
# std_pred: [B, pred_len, N] 不确定性
# samples: [100, B, pred_len, N] 所有样本
```

---

## 10. 参考文献

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
2. **DDIM**: Song et al., "Denoising Diffusion Implicit Models", ICLR 2021
3. **iTransformer**: Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024
4. **Diffusion for Time Series**: Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", ICML 2021
5. **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018

---

## 附录

### A. 文件结构

```
Time-Series-Library/
├── models/
│   ├── iTransformerDiffusionDirect.py  # 本模型
│   ├── iTransformerDiffusion.py        # ε-prediction 版本
│   └── GaussianDiffusion.py            # 基础扩散类
├── layers/
│   ├── Diffusion_layers.py             # UNet1D, FiLM 等
│   └── Embed.py                        # DataEmbedding_inverted
├── exp/
│   └── exp_diffusion_forecast.py       # 两阶段训练
└── scripts/
    └── diffusion_forecast/             # 运行脚本
```

### B. 模型参数量

```
iTransformerDiffusionDirect (ETTh1, d_model=128):
├── enc_embedding:     ~100K params
├── encoder:           ~200K params
├── projection:        ~10K params
├── denoise_net:       ~9M params
└── Total:             ~9.3M params

Checkpoint size: ~38 MB
```

### C. 显存占用

| 配置 | 训练显存 | 推理显存 (n=100) |
|------|----------|------------------|
| batch_size=32, FP32 | ~8 GB | ~6 GB |
| batch_size=32, AMP | ~5 GB | ~4 GB |
| batch_size=16, AMP | ~3 GB | ~3 GB |

---

*文档版本: 1.0 | 最后更新: 2025-01*
