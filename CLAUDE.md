# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Time-Series-Library (TSLib) is an open-source deep learning library from THU-ML supporting 40+ models for time series forecasting, imputation, anomaly detection, and classification. All experiments run through a unified CLI interface.

**本项目重点研究模型**: iTransformerDiffusion - 将 iTransformer 作为 backbone 的条件残差扩散模型 (CRD-Net)。

## Running Experiments

```bash
conda activate tslib

# === iTransformerDiffusion (概率预测, 两阶段训练) ===
python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model iTransformerDiffusion \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 96 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 128 --d_ff 128 \
  --diffusion_steps 1000 --beta_schedule cosine \
  --stage1_epochs 30 --stage2_epochs 20 \
  --n_samples 100 --use_amp

# 低显存运行 (8GB GPU)
bash scripts/diffusion_forecast/ETT_script/iTransformerDiffusion_ETTh1_8GB.sh

# 标准 iTransformer (确定性预测)
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model iTransformer \
  --data ETTh1 ...

# Test only (no training)
python run.py --is_training 0 [same args...]
```

**Task types:** `long_term_forecast`, `short_term_forecast`, `imputation`, `anomaly_detection`, `classification`, `zero_shot_forecast`, `diffusion_forecast`

**Key parameters:**
- `--seq_len`: Input sequence length (default 96)
- `--pred_len`: Prediction horizon (default 96)
- `--label_len`: Decoder start token length (default 48)
- `--enc_in/dec_in/c_out`: Number of variates (channels)
- `--d_model/d_ff`: Model dimensions
- `--e_layers/d_layers`: Encoder/decoder layers
- `--features`: M (multivariate→multivariate), S (univariate), MS (multivariate→univariate)

## Architecture

```
run.py                    # Entry point - parses args, routes to Exp classes
├── exp/
│   ├── exp_basic.py      # Model registry (model_dict) and base class
│   ├── exp_long_term_forecasting.py
│   ├── exp_diffusion_forecast.py   # ★ 扩散模型两阶段训练
│   └── ...
├── models/
│   ├── iTransformer.py             # 基础 iTransformer
│   ├── iTransformerDiffusion.py    # ★ iTransformer + CRD-Net 混合架构
│   ├── GaussianDiffusion.py        # 基础高斯扩散工具类
│   └── ...
├── layers/
│   ├── Embed.py                    # DataEmbedding_inverted
│   ├── Diffusion_layers.py         # ★ UNet1D, FiLM, VariateCrossAttention
│   ├── SelfAttention_Family.py
│   └── Transformer_EncDec.py
├── data_provider/
└── scripts/
    └── diffusion_forecast/         # ★ 扩散预测脚本
```

---

## iTransformerDiffusion Architecture (核心研究模型)

**设计理念**: 结合 iTransformer 的变量级注意力机制与条件残差扩散 (CRD-Net)，实现概率时序预测。

### 整体数据流

```
Input x_hist [B, seq_len, N]
    │
    ▼ iTransformer Backbone
┌────────────────────────────────────────┐
│  Instance Norm → DataEmbedding_inverted │
│  → Encoder (attention across variates)  │
│  → Projection                           │
└────────────────────────────────────────┘
    │                    │
    ▼                    ▼
y_det [B, pred_len, N]   z [B, N, d_model]  (encoder features)
    │                    │
    ▼                    │
Residual = y_true - y_det│    (训练时)
    │                    │
    ▼                    ▼
┌────────────────────────────────────────┐
│           CRD-Net (1D U-Net)            │
│  ┌──────────────────────────────────┐  │
│  │ Time Embedding (SinusoidalPosEmb) │  │
│  └──────────────────────────────────┘  │
│  ConditionProjector: z + t_emb → cond   │
│  ┌──────────────────────────────────┐  │
│  │   Encoder: DownBlocks + FiLM     │  │
│  │   Bottleneck: ResBlock + CrossAttn│  │
│  │   Decoder: UpBlocks + FiLM + XAttn│  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
    │
    ▼
y_final = y_det + sampled_residual  (推理时)
```

### 核心组件 (`layers/Diffusion_layers.py`)

| 组件 | 作用 |
|------|------|
| `SinusoidalPosEmb` | 扩散时间步的正弦位置编码 |
| `ConditionProjector` | 融合 iTransformer 特征 z 与时间嵌入 t_emb → 全局条件向量 |
| `FiLMLayer` | Feature-wise Linear Modulation: γ*h + β |
| `VariateCrossAttention` | 变量级交叉注意力，去噪特征 attend to 编码器特征 |
| `ResBlock1D` | 1D 残差块，带扩张卷积 + FiLM 调制 |
| `DownBlock1D` / `UpBlock1D` | U-Net 的下/上采样块 |
| `UNet1D` | 完整的 1D U-Net 去噪网络 |
| `ResidualNormalizer` | 残差归一化，EMA 跟踪统计量 |

### 两阶段训练策略 (`exp/exp_diffusion_forecast.py`)

```
Stage 1 (Warmup): 30 epochs
  ├── 训练: enc_embedding + encoder + projection
  ├── 损失: MSE(y_det, y_true)
  └── 学习率: 1e-4

Stage 2 (Joint): 20 epochs
  ├── 冻结: enc_embedding + encoder
  ├── 训练: projection + denoise_net + residual_normalizer
  ├── 损失: λ*MSE + (1-λ)*Diffusion (λ=0.5)
  └── 学习率: projection 1e-5, diffusion 1e-4
```

### 扩散过程

**前向扩散 (训练)**:
```python
# 计算残差并归一化
residual = y_true - y_det.detach()
residual_norm = residual_normalizer.normalize(residual)

# 加噪
t = random(0, timesteps)
xt = sqrt(ᾱt)*x0 + sqrt(1-ᾱt)*ε

# 预测噪声
noise_pred = denoise_net(xt, t, z)
loss_diff = MSE(noise_pred, noise)
```

**逆向采样 (推理)**:
- DDPM: 1000 步完整采样
- DDIM: 50 步加速采样 (η=0 确定性, η>0 随机性)
- 批量采样: `sample_ddpm_batch()` / `sample_ddim_batch()` 并行处理多样本
- 分块采样: `sample_chunked()` 控制显存使用

### 关键配置参数

```bash
# 扩散模型参数
--diffusion_steps 1000     # 扩散步数
--beta_schedule cosine     # beta 调度: linear/cosine
--cond_dim 256             # FiLM 条件维度

# 训练参数
--stage1_epochs 30         # Stage 1 轮数
--stage2_epochs 20         # Stage 2 轮数
--stage1_lr 1e-4           # Stage 1 学习率
--stage2_lr 1e-5           # Stage 2 学习率
--loss_lambda 0.5          # MSE 损失权重

# 采样参数
--n_samples 100            # 概率预测采样数
--use_ddim                 # 使用 DDIM 加速采样
--ddim_steps 50            # DDIM 步数
--chunk_size 10            # 分块采样大小 (控制显存)
--use_amp                  # 启用混合精度 (节省 30-50% 显存)
```

### 评估指标

**点预测**: MSE, MAE, RMSE
**概率预测**: CRPS (Continuous Ranked Probability Score), Calibration (50%/90% 覆盖率), Sharpness

---

## 基础 iTransformer Architecture

**Paper:** https://arxiv.org/abs/2310.06625 (ICLR 2024)

iTransformer inverts the standard Transformer by applying self-attention across **variates (channels)** instead of the temporal dimension. This is a lightweight, encoder-only architecture.

**Data flow:**
```
Input [B, seq_len, variates]
  → Normalize per variate
  → Permute to [B, variates, seq_len]
  → Linear(seq_len → d_model) → [B, variates, d_model]
  → Encoder (attention across variates)
  → Linear(d_model → pred_len) → [B, variates, pred_len]
  → Permute back, denormalize
Output [B, pred_len, variates]
```

**iTransformer-specific settings:**
- Uses smaller `d_model=128, d_ff=128` (vs default 512/2048)
- Typically 2 encoder layers
- No decoder needed (encoder-only)

---

## Testing

```bash
# 运行 iTransformerDiffusion 单元测试
cd ~/projects/Time-Series-Library
python -m pytest tests/test_iTransformerDiffusion.py -v

# 边界情况测试
python -m pytest tests/test_iTransformerDiffusion_edge_cases.py -v
```

## Adding a New Model

1. Create `models/YourModel.py` with `class Model(nn.Module)` taking `configs` arg
2. Import and add to `model_dict` in `exp/exp_basic.py`
3. Create run scripts in `scripts/<task>/<dataset>/YourModel.sh`

## Data

Datasets go in `./dataset/`. Common ones: ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic.

ETT datasets have 7 variates. Set `--enc_in 7 --dec_in 7 --c_out 7`.

## Key Files for iTransformerDiffusion Development

| 文件 | 描述 |
|------|------|
| `models/iTransformerDiffusion.py` | ★ 主模型实现 (backbone + CRD-Net) |
| `layers/Diffusion_layers.py` | ★ 扩散组件 (UNet1D, FiLM, CrossAttn) |
| `exp/exp_diffusion_forecast.py` | ★ 两阶段训练逻辑 |
| `models/GaussianDiffusion.py` | 基础高斯扩散工具类 |
| `models/iTransformer.py` | 基础 iTransformer 参考 |
| `layers/Embed.py:129-143` | `DataEmbedding_inverted` |
| `scripts/diffusion_forecast/` | 扩散预测脚本 |
| `tests/test_iTransformerDiffusion.py` | 单元测试 |
