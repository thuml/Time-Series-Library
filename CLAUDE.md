# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Time Series Library (TSLib) is a deep learning benchmark for time series analysis covering five tasks: long-term forecasting, short-term forecasting, imputation, anomaly detection, and classification. It includes 30+ models from TimesNet to Large Time Series Models (LTSMs) like Chronos and Moirai.

## Commands

### Environment Setup
```bash
conda create -n tslib python=3.11
conda activate tslib
pip install -r requirements.txt
```

### Training & Evaluation
```bash
# Single run with specific parameters
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model TimesNet \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7

# Run experiment script (recommended)
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
```

### Task-Specific Commands
```bash
# Long-term forecasting
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh

# Short-term forecasting (M4)
bash ./scripts/short_term_forecast/TimesNet_M4.sh

# Imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh

# Anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh

# Classification
bash ./scripts/classification/TimesNet.sh

# Zero-shot forecasting (LTSMs)
bash ./scripts/long_term_forecast/ETT_script/LTSM.sh
```

### Key Arguments
- `--task_name`: `long_term_forecast`, `short_term_forecast`, `imputation`, `anomaly_detection`, `classification`, `zero_shot_forecast`
- `--is_training`: `1` for train+test, `0` for test only
- `--model`: Model name matching key in `exp/exp_basic.py:model_dict`
- `--data`: Dataset key from `data_provider/data_factory.py:data_dict` (ETTh1, ETTh2, ETTm1, ETTm2, custom, m4, PSM, MSL, SMAP, SMD, SWAT, UEA)
- `--features`: `M` (multivariate→multivariate), `S` (univariate→univariate), `MS` (multivariate→univariate)
- `--seq_len`: Input sequence length
- `--pred_len`: Prediction horizon
- `--enc_in`, `--dec_in`, `--c_out`: Number of input/output variables

## Architecture

### Entry Point Flow
`scripts/*.sh` → `run.py` → `Exp_*` class (via `task_name`) → `data_provider` + `models` → checkpoints saved to `./checkpoints/`

### Key Directories
- **`exp/`**: Task pipelines. `Exp_Basic` registers models; subclasses (`Exp_Long_Term_Forecast`, etc.) implement `_get_data()`, `train()`, `test()`
- **`models/`**: Model architectures. Each file is self-contained. Add new models here and register in `Exp_Basic.model_dict`
- **`layers/`**: Reusable components (attention mechanisms, embeddings, decomposition blocks) shared across models
- **`data_provider/`**: Dataset loaders. `data_factory.py` maps dataset names to loader classes; `data_loader.py` handles windowing
- **`scripts/`**: Bash scripts with paper configurations organized by task/dataset/model
- **`utils/`**: Metrics (`metrics.py`), early stopping (`tools.py`), augmentations, masking

### Adding a New Model
1. Create `models/YourModel.py` following `models/Transformer.py` pattern
2. Add to imports and `model_dict` in `exp/exp_basic.py`
3. Create scripts under `scripts/<task>/<dataset>/YourModel.sh`

### Dataset Structure
Place datasets in `./dataset/`. Expected structure per README:
- ETT: `./dataset/ETT-small/ETTh1.csv`
- Anomaly detection: `./dataset/PSM/`, `./dataset/MSL/`, etc.
- Classification: `./dataset/Heartbeat/`

## Model Categories

**Forecasting**: TimesNet, iTransformer, PatchTST, TimeMixer, TimeXer, DLinear, Autoformer, Informer, FEDformer, Transformer

**LTSMs (zero-shot)**: Chronos, Chronos2, Moirai, TimesFM, TimeMoE, Sundial, TiRex

**Anomaly Detection**: TimesNet, KANAD (KAN-AD)

**Special Requirements**:
- Mamba: requires `mamba_ssm` package (CUDA-specific wheel)
- Moirai: requires `pip install uni2ts --no-deps`

## Core Model Designs

### TimesNet (ICLR 2023)
**Core Innovation**: Temporal 2D-Variation Modeling
- Uses FFT to discover dominant periods in time series
- Reshapes 1D sequence into 2D tensor by period (e.g., period=24: [1,168] → [7,24])
- Applies 2D convolution (Inception blocks) to capture intra-period and inter-period variations
- Aggregates multiple periods with FFT amplitude weights
- Unified framework supporting all 5 tasks

### iTransformer (ICLR 2024)
**Core Innovation**: Inverted Transformer Architecture
- Traditional: attention across time steps
- iTransformer: attention across variables (channels)
- Each variable's entire time series becomes one token: [B,L,N] → [B,N,D]
- Better captures multivariate dependencies
- Avoids attention dilution over long sequences

### PatchTST (ICLR 2023)
**Core Innovation**: Patching + Channel Independence
- Splits time series into fixed-length patches (like ViT for images)
- Each variable processed independently, sharing Transformer weights
- Reduces sequence length, lowering computational cost
- Key params: `patch_len=16`, `stride=8` (50% overlap)

### Autoformer (NeurIPS 2021)
**Core Innovation**: Series Decomposition + Auto-Correlation
- Decomposes series into trend + seasonal components using moving average
- Auto-Correlation mechanism replaces self-attention
- Exploits periodicity for information aggregation
- O(L log L) complexity

### Informer (AAAI 2021 Best Paper)
**Core Innovation**: ProbSparse Attention + Distilling
- ProbSparse attention: selects important queries via KL divergence, O(L log L)
- Self-attention distilling: halves sequence length per layer via convolution
- Generative decoder: outputs entire prediction at once (non-autoregressive)

### DLinear (AAAI 2023)
**Core Innovation**: Minimalist Linear Model
- Questions Transformer effectiveness for time series
- Simple decomposition + two linear layers (trend + seasonal)
- Minimal parameters, fast training
- `individual=True`: separate linear layer per variable
