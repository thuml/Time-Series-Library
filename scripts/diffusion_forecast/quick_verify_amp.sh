#!/bin/bash

# Quick test script to verify AMP and batch sampling optimizations
# This runs a minimal training to check everything works

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Quick Verification Test"
echo "- AMP training enabled"
echo "- Batch sampling with chunk_size=5"
echo "=========================================="

python -u run.py \
    --task_name diffusion_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_quick_amp_test \
    --model iTransformerDiffusion \
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
    --d_model 128 \
    --d_ff 128 \
    --n_heads 8 \
    --dropout 0.1 \
    --diffusion_steps 100 \
    --beta_schedule cosine \
    --cond_dim 256 \
    --stage1_epochs 2 \
    --stage2_epochs 2 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --patience 3 \
    --n_samples 10 \
    --chunk_size 5 \
    --use_ddim \
    --ddim_steps 10 \
    --use_amp \
    --itr 1

echo ""
echo "=========================================="
echo "Verification complete!"
echo "=========================================="
