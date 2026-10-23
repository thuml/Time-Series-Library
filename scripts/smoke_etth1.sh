#!/usr/bin/env bash
# Real ETTh1 / real CEEMDAN integration check. Limited runs are not benchmarks.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-CEEMDANPatchTST}"
"$PYTHON" -u run_longExp.py \
  --is_training 1 --model_id smoke --model "$MODEL" \
  --data ETTh1 --root_path ./data --data_path ETTh1.csv \
  --features S --target OT --enc_in 1 --dec_in 1 --c_out 1 \
  --seq_len 96 --label_len 48 --pred_len 24 --patch_len 16 --stride 8 \
  --e_layers 2 --d_model 8 --n_heads 2 --d_ff 16 \
  --dropout 0 --fc_dropout 0 --head_dropout 0 --affine 1 \
  --progressive_decomp ceemdan --fusion_attention hybrid \
  --ceemdan_trials 2 --ceemdan_max_imf 2 --ceemdan_max_sift 20 \
  --train_epochs 2 --patience 2 --batch_size 2 --learning_rate 0.0001 \
  --lradj TST --pct_start 0.3 --itr 1 --num_workers 0 \
  --max_train_steps 2 --max_eval_steps 1 --des integration "$@"
