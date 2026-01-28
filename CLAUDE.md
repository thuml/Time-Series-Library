# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 交互要求

- **Thinking 思考过程用中文表述**
- **Reply 回答也要用中文回复**
- **代码注释用中文编写**

## Overview

Time-Series-Library (TSLib) is an open-source deep learning library from THU-ML supporting 40+ models for time series forecasting, imputation, anomaly detection, and classification. All experiments run through a unified CLI interface.

**本项目重点研究模型**:
- **iTransformerDiffusion** - 条件残差扩散模型 (CRD-Net)
- **iTransformerDiffusionDirect** - 直接预测扩散模型（支持 x₀/ε/v 多种参数化）

## 环境配置

### 依赖安装

```bash
conda create -n tslib python=3.9
conda activate tslib
pip install -r requirements.txt
```

**核心依赖**:
- torch==2.5.1 (需要 CUDA 支持)
- einops==0.8.1
- scikit-learn==1.2.2
- scipy==1.10.1
- tqdm==4.64.1

**可选依赖**:
- mamba_ssm (用于 Mamba 模型)
- transformers (用于预训练模型)
- datasets (用于数据加载)

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

### 模型选择指南

**何时使用 iTransformerDiffusion**:
- 需要高质量的确定性预测 + 不确定性量化
- 已有强大的确定性 backbone
- 残差建模更合适的场景
- 两阶段训练策略

**何时使用 iTransformerDiffusionDirect**:
- 端到端概率建模
- 需要更稳定的训练（v-prediction）
- 探索不同参数化策略（x₀/ε/v）
- 更简洁的架构

**何时使用基础 iTransformer**:
- 只需要点预测，不需要不确定性量化
- 训练/推理速度优先
- 资源受限环境

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

## iTransformerDiffusionDirect Architecture (直接预测变体)

**设计理念**: 直接预测目标而非残差，支持多种参数化策略（x₀/ε/v），训练更稳定。

### 与 iTransformerDiffusion 的对比

| 特性 | iTransformerDiffusion | iTransformerDiffusionDirect |
|------|----------------------|----------------------------|
| **预测目标** | 残差 (y_true - y_det) | 直接预测 y_true |
| **参数化** | 单一 (噪声预测) | 多种 (x₀/ε/v) |
| **训练稳定性** | 需要残差归一化 | v-prediction 最稳定 |
| **训练模式** | 两阶段分离训练 | 端到端或两阶段 |
| **适用场景** | 确定性 backbone 强 | 端到端概率建模 |

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
y_det [B, pred_len, N]   z [B, N, d_model]  (条件特征)
    │                    │
    ▼                    ▼
目标 y_true              1D U-Net Denoiser
    │                   (FiLM + CrossAttention)
    │                    │
    └────────────────────┘
            │
            ▼
    直接预测 y_true (训练)
    概率采样 (推理)
```

### 参数化策略

**v-prediction (推荐)** ✅
- 所有时间步信噪比平衡
- 无需 clamp() 稳定预测
- 更好的梯度流动
- 数学定义: v = √ᾱ_t · ε − √(1-ᾱ_t) · x₀

**x₀-prediction** 🟡
- 直接预测目标，直观易懂
- 需要 clamp() 防止数值不稳定
- 早期时间步信噪比低

**ε-prediction** 🔴
- DDPM 标准方法
- 后期时间步信噪比低
- 训练不够稳定

### 快速运行

```bash
# 推荐配置 (v-prediction, 端到端训练)
python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model iTransformerDiffusionDirect \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 96 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --d_ff 64 \
  --parameterization v \
  --training_mode end_to_end \
  --train_epochs 50 \
  --n_samples 100 \
  --use_amp

# 低显存版本 (8GB GPU)
bash scripts/diffusion_forecast/ETT_script/iTransformerDiffusionDirect_ETTh1_v1.sh

# 两阶段训练模式
python run.py \
  --model iTransformerDiffusionDirect \
  --training_mode two_stage \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --parameterization v \
  [其他参数...]
```

### 关键配置参数

```bash
# 参数化选择
--parameterization v           # v/x0/epsilon (推荐 v)

# 训练模式
--training_mode end_to_end     # end_to_end/two_stage
--train_epochs 50              # 端到端训练轮数
--warmup_epochs 10             # 预热轮数

# 扩散参数（与 iTransformerDiffusion 相同）
--diffusion_steps 1000
--beta_schedule cosine
--cond_dim 256
```

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

# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_iTransformerDiffusion.py::test_forward -v
```

## 结果分析与调试

### 查看训练结果

```bash
# 查看所有实验结果
ls results/diffusion_forecast/

# 查看特定实验的日志
tail -f checkpoints/<experiment_name>/log.txt

# 查看最新实验日志
tail -f checkpoints/$(ls -t checkpoints/ | head -1)/log.txt

# 查看测试结果
ls test_results/diffusion_forecast/

# 查看特定结果文件
cat results/diffusion_forecast/result_<model>_<data>_<seq>_<pred>.txt
```

### 性能监控

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控训练进度（使用项目内置脚本）
bash scripts/phase2_monitor.sh

# 查看实时日志
tail -f logs/*.log
```

### 调试模式

```bash
# 快速调试（小数据集，少轮数）
python run.py \
  --task_name diffusion_forecast \
  --model iTransformerDiffusion \
  --data ETTh1 \
  --train_epochs 2 \
  --stage1_epochs 1 \
  --stage2_epochs 1 \
  --n_samples 10 \
  --batch_size 4

# 过拟合单个 batch（验证模型实现正确性）
python run.py \
  --task_name diffusion_forecast \
  --model iTransformerDiffusion \
  --data ETTh1 \
  --train_epochs 100 \
  --batch_size 1 \
  --num_workers 0
```

### 常见问题排查

**显存不足 (OOM)**:
```bash
# 解决方案 1: 启用混合精度 + 减小 batch size
--use_amp --batch_size 8

# 解决方案 2: 分块采样
--chunk_size 5 --n_samples 50

# 解决方案 3: 减小模型尺寸
--d_model 64 --d_ff 64 --unet_channels [32,64,128,256]
```

**训练不稳定**:
```bash
# 解决方案 1: 使用 v-prediction (仅 Direct 模型)
--parameterization v

# 解决方案 2: 降低学习率
--learning_rate 5e-5 --stage2_lr 5e-6

# 解决方案 3: 增加预热轮数
--warmup_epochs 20
```

**推理速度慢**:
```bash
# 解决方案: DDIM 加速采样
--use_ddim --ddim_steps 20 --n_samples 50
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
| `models/iTransformerDiffusionDirect.py` | ★ 直接预测变体 (支持 x₀/ε/v) |
| `layers/Diffusion_layers.py` | ★ 扩散组件 (UNet1D, FiLM, CrossAttn) |
| `exp/exp_diffusion_forecast.py` | ★ 两阶段训练逻辑 |
| `models/GaussianDiffusion.py` | 基础高斯扩散工具类 |
| `models/iTransformer.py` | 基础 iTransformer 参考 |
| `layers/Embed.py:129-143` | `DataEmbedding_inverted` |
| `scripts/diffusion_forecast/` | 扩散预测脚本 |
| `tests/test_iTransformerDiffusion.py` | 单元测试 |

---

## 常见开发工作流

### 1. 实验新模型变体

```bash
# 步骤 1: 复制基础模型
cp models/iTransformer.py models/MyModel.py

# 步骤 2: 修改模型（确保类名为 Model）
# 编辑 models/MyModel.py

# 步骤 3: 注册模型
# 在 exp/exp_basic.py 的 model_dict 中添加:
# 'MyModel': MyModel,

# 步骤 4: 创建运行脚本
cp scripts/long_term_forecast/ETT_script/iTransformer.sh \
   scripts/long_term_forecast/ETT_script/MyModel.sh

# 步骤 5: 运行测试
bash scripts/long_term_forecast/ETT_script/MyModel.sh
```

### 2. 批量实验

```bash
# 启动多个实验（后台运行）
bash scripts/phase2_launch_all.sh

# 监控实验进度
bash scripts/phase2_monitor.sh

# 收集所有结果
bash scripts/phase2_collect_results.sh

# 分析结果
python scripts/analyze_prediction_gap.py
```

### 3. 模型对比评估

```bash
# 运行基线模型（确定性预测）
python run.py \
  --task_name long_term_forecast \
  --model iTransformer \
  --data ETTh1 \
  --seq_len 96 --pred_len 96

# 运行扩散模型（概率预测）
python run.py \
  --task_name diffusion_forecast \
  --model iTransformerDiffusion \
  --data ETTh1 \
  --seq_len 96 --pred_len 96 \
  --n_samples 100

# 对比结果
# 查看 results/ 和 test_results/ 目录下的 .txt 文件
```

### 4. 超参数调优

```bash
# 学习率调优
for lr in 1e-3 5e-4 1e-4 5e-5; do
  python run.py --learning_rate $lr [其他参数...]
done

# 模型尺寸调优
for dim in 64 128 256; do
  python run.py --d_model $dim --d_ff $dim [其他参数...]
done

# 扩散步数调优
for steps in 100 500 1000; do
  python run.py --diffusion_steps $steps [其他参数...]
done
```

### 5. 跨数据集评估

```bash
# ETT 系列
for data in ETTh1 ETTh2 ETTm1 ETTm2; do
  python run.py --data $data --root_path ./dataset/ETT-small/ \
    --data_path ${data}.csv --enc_in 7 --dec_in 7 --c_out 7 \
    [其他参数...]
done

# 大规模数据集
for data in Weather ECL Traffic; do
  python run.py --data $data --root_path ./dataset/ \
    [调整 enc_in/dec_in/c_out...] [其他参数...]
done
```

---

## 项目文档索引

| 文档 | 描述 |
|------|------|
| `README.md` | iTransformerDiffusionDirect 模型说明（主README） |
| `CLAUDE.md` | Claude Code 操作指南（本文件） |
| `docs/iTransformerDiffusionDirect_Technical_Document.md` | 技术文档详解 |
| `docs/iTransformerDiffusionDirect_Technical_Doc.md` | 技术文档（简版） |
| `docs/iTransformerDiffusionDirect_Refactoring_Plan.md` | 重构计划 |
| `docs/FR2_INTEGRATION_GUIDE.md` | 特征重构集成指南 |
| `tests/TEST_SUMMARY.md` | 测试总结 |
| `HOW_TO_USE_BEST_MODEL.md` | 最佳模型使用指南 |
| `IMPLEMENTATION_SUMMARY.md` | 实现总结 |
| `CONTRIBUTING.md` | 贡献指南 |

---

## 相关论文

- **iTransformer**: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625) (ICLR 2024)
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (NeurIPS 2020)
- **DDIM**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (ICLR 2021)
- **v-Prediction**: [Progressive Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512) (ICLR 2022)
- **Diffusion for Time Series**: Multiple recent works on probabilistic forecasting
