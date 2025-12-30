# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Time-Series-Library (TSLib) is an open-source deep learning library from THU-ML supporting 40+ models for time series forecasting, imputation, anomaly detection, and classification. All experiments run through a unified CLI interface.

## Running Experiments

```bash
conda activate tslib

# Basic experiment pattern
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model iTransformer \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 96 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7

# Test only (no training)
python run.py --is_training 0 [same args...]

# Pre-made scripts
bash scripts/long_term_forecast/ETT_script/iTransformer_ETTh2.sh
```

**Task types:** `long_term_forecast`, `short_term_forecast`, `imputation`, `anomaly_detection`, `classification`, `zero_shot_forecast`

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
│   ├── exp_imputation.py
│   └── ...
├── models/               # 40+ model implementations, each with class Model(nn.Module)
│   ├── iTransformer.py
│   ├── TimesNet.py
│   └── ...
├── layers/               # Shared components (attention, embeddings, encoders)
│   ├── Embed.py          # DataEmbedding, DataEmbedding_inverted, PatchEmbedding
│   ├── SelfAttention_Family.py
│   └── Transformer_EncDec.py
├── data_provider/        # Dataset loaders (ETT, M4, anomaly datasets, UEA)
└── scripts/              # Pre-configured experiment scripts by task/dataset
```

## iTransformer Architecture

**Paper:** https://arxiv.org/abs/2310.06625 (ICLR 2024)

iTransformer inverts the standard Transformer by applying self-attention across **variates (channels)** instead of the temporal dimension. This is a lightweight, encoder-only architecture.

**Key components in `models/iTransformer.py`:**

1. **DataEmbedding_inverted** (`layers/Embed.py:129-143`): Permutes input from `[B, T, N]` to `[B, N, T]`, then projects time dimension to `d_model`
2. **Encoder stack**: Standard Transformer encoder with FullAttention, but attention operates across variates
3. **Instance normalization**: Per-variate mean/std normalization before processing, denormalization after

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

## Adding a New Model

1. Create `models/YourModel.py` with `class Model(nn.Module)` taking `configs` arg
2. Import and add to `model_dict` in `exp/exp_basic.py`
3. Create run scripts in `scripts/<task>/<dataset>/YourModel.sh`

## Data

Datasets go in `./dataset/`. Common ones: ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic.

ETT datasets have 7 variates. Set `--enc_in 7 --dec_in 7 --c_out 7`.

## Key Files for iTransformer Development

- `models/iTransformer.py` - Model implementation
- `layers/Embed.py` - `DataEmbedding_inverted` class (lines 129-143)
- `layers/SelfAttention_Family.py` - `FullAttention` mechanism
- `layers/Transformer_EncDec.py` - `Encoder`, `EncoderLayer`
- `scripts/long_term_forecast/*/iTransformer*.sh` - Run scripts
