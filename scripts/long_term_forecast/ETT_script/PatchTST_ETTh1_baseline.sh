#!/bin/bash

# Phase 2.2: 横向对比 - PatchTST
# ETTh1 96→96 预测任务
# 配置: e_layers=1, 30 epochs

model_name=PatchTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_baseline \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Phase2_Baseline' \
  --n_heads 2 \
  --train_epochs 30 \
  --patience 5 \
  --batch_size 32 \
  --itr 1

echo "PatchTST 基线训练完成！"
