#!/bin/bash

# iTransformerDiffusion on ETTh1 - 8GB VRAM Optimized
# This script is optimized for GPUs with limited memory (8GB)
#
# Key optimizations:
# - AMP (FP16) training: ~30-50% memory savings
# - Reduced batch size: 32 -> 8
# - Reduced n_samples: 100 -> 50
# - Batch sampling with chunk_size=5
# - Reduced DDIM steps: 50 -> 20

export CUDA_VISIBLE_DEVICES=0

# Base settings
model_name=iTransformerDiffusion
data=ETTh1
root_path=./dataset/ETT-small/
data_path=ETTh1.csv

# Common parameters
seq_len=96
label_len=48
enc_in=7
dec_in=7
c_out=7
d_model=128
d_ff=128
n_heads=8
e_layers=2
d_layers=1
dropout=0.1

# Diffusion parameters (8GB optimized)
diffusion_steps=1000
beta_schedule=cosine
cond_dim=256

# Training parameters (8GB optimized)
batch_size=8            # Reduced from 32
stage1_epochs=30
stage2_epochs=20
learning_rate=0.0001
patience=5

# Sampling parameters (8GB optimized)
n_samples=50            # Reduced from 100
chunk_size=5            # Conservative chunk size
ddim_steps=20           # Reduced from 50

echo "=========================================="
echo "iTransformerDiffusion - 8GB VRAM Optimized"
echo "Dataset: $data"
echo "=========================================="

for pred_len in 96 192 336 720
do
    echo ""
    echo ">>> Training pred_len=$pred_len"
    echo ""

    python -u run.py \
        --task_name diffusion_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ${data}_${seq_len}_${pred_len} \
        --model $model_name \
        --data $data \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers $e_layers \
        --d_layers $d_layers \
        --factor 3 \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --d_model $d_model \
        --d_ff $d_ff \
        --n_heads $n_heads \
        --dropout $dropout \
        --diffusion_steps $diffusion_steps \
        --beta_schedule $beta_schedule \
        --cond_dim $cond_dim \
        --stage1_epochs $stage1_epochs \
        --stage2_epochs $stage2_epochs \
        --stage1_lr $learning_rate \
        --stage2_lr 1e-5 \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --train_epochs 10 \
        --patience $patience \
        --n_samples $n_samples \
        --chunk_size $chunk_size \
        --use_ddim \
        --ddim_steps $ddim_steps \
        --use_amp \
        --itr 1
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
