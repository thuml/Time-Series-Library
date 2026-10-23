#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
FEATURES="${FEATURES:-MS}"
PRED_LEN="${PRED_LEN:-96}"
CHANNELS=862
OUTPUTS=1
if [[ "$FEATURES" == M ]]; then OUTPUTS=862; fi
if [[ "$FEATURES" == S ]]; then CHANNELS=1; fi
"$PYTHON" scripts/download_traffic.py
"$PYTHON" -u run_longExp.py \
  --is_training 1 --model_id "Traffic_${FEATURES}_96_${PRED_LEN}" --model TimeXer \
  --data traffic --features "$FEATURES" --target OT --freq h \
  --enc_in "$CHANNELS" --dec_in "$CHANNELS" --c_out "$OUTPUTS" \
  --seq_len 96 --label_len 48 --pred_len "$PRED_LEN" --patch_len 16 --use_norm 1 \
  --d_model 128 --n_heads 8 --e_layers 2 --d_ff 256 --dropout 0.1 --activation gelu \
  --batch_size 16 --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --lradj type1 --itr 1 --num_workers 0 --des timexer "$@"
