#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-CEEMDANPatchTST}"
# Progressive CEEMDAN is CPU-bound. This starts a full training run.
"$PYTHON" -u run_longExp.py \
  --is_training 1 --model_id ETTh1_96_96 --model "$MODEL" \
  --data ETTh1 --root_path ./data --data_path ETTh1.csv \
  --features M --target OT --enc_in 7 --dec_in 7 --c_out 7 \
  --seq_len 96 --label_len 48 --pred_len 96 --patch_len 16 --stride 8 \
  --e_layers 2 --d_model 16 --n_heads 4 --d_ff 64 \
  --dropout 0.1 --fc_dropout 0 --head_dropout 0 --affine 1 \
  --progressive_decomp ceemdan --fusion_attention hybrid \
  --ceemdan_trials 8 --ceemdan_max_imf 3 --ceemdan_max_sift 50 \
  --train_epochs 30 --patience 5 --batch_size 4 --learning_rate 0.0001 \
  --lradj TST --pct_start 0.3 --itr 1 --num_workers 0 --des progressive "$@"
