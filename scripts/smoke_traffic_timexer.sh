#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Exercises real Traffic through train, validation, checkpoint reload, test and prediction.
bash scripts/train_traffic_timexer.sh \
  --model_id smoke_traffic_timexer --d_model 16 --n_heads 4 --d_ff 32 \
  --batch_size 2 --train_epochs 1 --max_train_steps 2 --max_eval_steps 2 \
  --checkpoint_interval 0 --do_predict "$@"
