#!/bin/bash
# Quick test script for iTransformerDiffusion
# Uses minimal epochs for fast verification

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformerDiffusion

echo "========================================"
echo "Quick test: iTransformerDiffusion on ETTh1"
echo "========================================"

python -u run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_test \
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
  --des 'QuickTest' \
  --d_model 128 \
  --d_ff 128 \
  --n_heads 8 \
  --diffusion_steps 100 \
  --beta_schedule cosine \
  --cond_dim 128 \
  --stage1_epochs 2 \
  --stage2_epochs 2 \
  --stage1_lr 1e-4 \
  --stage2_lr 1e-5 \
  --n_samples 10 \
  --batch_size 16 \
  --patience 3 \
  --itr 1

echo "========================================"
echo "Test completed!"
echo "========================================"
