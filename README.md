# iTransformerDiffusion

**Probabilistic Time Series Forecasting with iTransformer + Conditional Residual Diffusion**

iTransformerDiffusion 是一个结合 iTransformer 和扩散模型的概率时间序列预测框架，能够同时提供点预测和不确定性量化。

<p align="center">
  <img src="./pic/architecture.png" width="800" alt="iTransformerDiffusion Architecture"/>
</p>

## Highlights

- **双重预测能力**: 同时输出确定性预测 (点估计) 和概率预测 (不确定性区间)
- **残差扩散设计**: 扩散模型在残差空间操作，更稳定且更快收敛
- **高效采样**: 支持 DDPM 和 DDIM 采样，批量并行加速 10-50x
- **内存优化**: 支持 AMP 混合精度训练，8GB 显存即可训练
- **完整评估指标**: 包含 CRPS、Calibration、Sharpness 等概率预测指标

## Architecture

```
Input x_hist [B, seq_len, N]
    │
    ▼
┌─────────────────────────────────┐
│     iTransformer Backbone       │
│  (Variate-level Self-Attention) │
└─────────────────────────────────┘
    │
    ├──► y_det [B, pred_len, N]  (Deterministic Prediction)
    │
    ▼
┌─────────────────────────────────┐
│   Conditional Residual          │
│   Diffusion (CRD-Net)           │
│   ┌───────────────────────────┐ │
│   │  1D U-Net + FiLM + CrossAttn│ │
│   └───────────────────────────┘ │
└─────────────────────────────────┘
    │
    ▼
y_final = y_det + residual  [B, pred_len, N]
    │
    ▼
Probabilistic Output: mean, std, samples
```

## Key Components

| Component | Description |
|-----------|-------------|
| **iTransformer Backbone** | 变量级注意力机制，捕获多变量间依赖关系 |
| **1D U-Net Denoiser** | 带 FiLM 条件调制的 U-Net，处理时序残差 |
| **Cross-Attention** | 允许去噪特征与编码器特征交互 |
| **Residual Normalizer** | EMA 统计量追踪，确保扩散输入稳定 |

## Installation

```bash
# Clone the repository
git clone https://github.com/Lfy181/iTransformerDiffusion.git
cd iTransformerDiffusion

# Create conda environment
conda create -n itdiff python=3.9
conda activate itdiff

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (recommended)

## Quick Start

### Training

```bash
# Standard training (requires ~16GB GPU memory)
bash scripts/diffusion_forecast/ETT_script/iTransformerDiffusion_ETTh1.sh

# Memory-efficient training (works with 8GB GPU)
bash scripts/diffusion_forecast/ETT_script/iTransformerDiffusion_ETTh1_8GB.sh

# Quick test
bash scripts/diffusion_forecast/quick_test.sh
```

### Basic Usage

```python
from models.iTransformerDiffusion import Model
import torch

# Configuration
class Config:
    task_name = 'long_term_forecast'
    seq_len = 96
    pred_len = 96
    enc_in = 7  # number of variates
    d_model = 128
    d_ff = 128
    n_heads = 4
    e_layers = 2
    dropout = 0.1
    embed = 'timeF'
    freq = 'h'
    factor = 1
    activation = 'gelu'
    # Diffusion configs
    diffusion_steps = 1000
    beta_schedule = 'cosine'
    n_samples = 100

# Initialize model
config = Config()
model = Model(config).cuda()

# Input data
x_enc = torch.randn(32, 96, 7).cuda()  # [B, seq_len, N]
x_mark_enc = torch.randn(32, 96, 4).cuda()  # [B, seq_len, M]

# Probabilistic prediction
mean_pred, std_pred, samples = model.predict(
    x_enc, x_mark_enc,
    n_samples=100,
    use_ddim=True,
    ddim_steps=50
)

# mean_pred: [B, pred_len, N] - point prediction
# std_pred: [B, pred_len, N] - uncertainty
# samples: [n_samples, B, pred_len, N] - all samples
```

## Training Strategy

iTransformerDiffusion 采用两阶段训练策略:

### Stage 1: Backbone Warmup (30 epochs)
- 只训练 iTransformer backbone
- 使用 MSE loss
- 学习率: 1e-4

### Stage 2: Joint Training (20 epochs)
- 冻结编码器，训练扩散组件
- 联合损失: λ·MSE + (1-λ)·Diffusion
- 学习率: 1e-5 (projection), 1e-4 (diffusion)

```bash
# Full training pipeline
python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model iTransformerDiffusion \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --use_amp  # Enable mixed precision
```

## Evaluation Metrics

### Point Prediction
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

### Probabilistic Prediction
- **CRPS** (Continuous Ranked Probability Score)
- **Calibration** (50%, 90% coverage)
- **Sharpness** (Average prediction interval width)

## Results

### ETTh1 Dataset (pred_len=96)

| Model | MSE | MAE | CRPS |
|-------|-----|-----|------|
| iTransformer | 0.386 | 0.405 | - |
| **iTransformerDiffusion** | **0.382** | **0.401** | **0.198** |

### Uncertainty Visualization

<p align="center">
  <img src="./pic/uncertainty.png" width="600" alt="Uncertainty Visualization"/>
</p>

## Project Structure

```
iTransformerDiffusion/
├── models/
│   ├── iTransformerDiffusion.py  # Main model
│   └── GaussianDiffusion.py      # Diffusion utilities
├── layers/
│   └── Diffusion_layers.py       # UNet1D, FiLM, CrossAttention
├── exp/
│   └── exp_diffusion_forecast.py # Two-stage training
├── scripts/
│   └── diffusion_forecast/       # Run scripts
├── tests/                        # Unit tests
└── docs/                         # Documentation
```

## Memory Optimization

对于显存有限的 GPU (8GB)，可以使用以下优化:

```bash
# Enable AMP (Automatic Mixed Precision)
python run.py --use_amp

# Reduce batch size and diffusion steps
python run.py --batch_size 8 --diffusion_steps 100

# Use DDIM for faster sampling
python run.py --use_ddim --ddim_steps 50
```

## Citation

如果这个项目对你有帮助，请引用:

```bibtex
@misc{iTransformerDiffusion2024,
  title={iTransformerDiffusion: Probabilistic Time Series Forecasting with Conditional Residual Diffusion},
  author={Lfy181},
  year={2024},
  howpublished={\url{https://github.com/Lfy181/iTransformerDiffusion}}
}
```

## Acknowledgement

本项目基于以下优秀工作:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) - THU-ML 时间序列分析库
- [iTransformer](https://arxiv.org/abs/2310.06625) - Inverted Transformers for Time Series
- [DDPM](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [DDIM](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models

## License

MIT License
