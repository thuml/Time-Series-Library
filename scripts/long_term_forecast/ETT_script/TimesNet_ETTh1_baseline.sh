#!/bin/bash

# Phase 2.2: 横向对比 - TimesNet
# ETTh1 96→96 预测任务
# 配置: d_model=16, e_layers=2, 30 epochs

model_name=TimesNet

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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Phase2_Baseline' \
  --top_k 5 \
  --train_epochs 30 \
  --patience 5 \
  --batch_size 32 \
  --itr 1

echo "TimesNet 基线训练完成！"
