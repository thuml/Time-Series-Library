#!/bin/bash
# iTransformerDiffusion on ETTh1 dataset
# Two-stage training: Stage 1 (backbone warmup) + Stage 2 (joint diffusion training)

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformerDiffusion

# ETTh1 96->96
python -u run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
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
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --n_heads 8 \
  --diffusion_steps 1000 \
  --beta_schedule cosine \
  --cond_dim 256 \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --stage1_lr 1e-4 \
  --stage2_lr 1e-5 \
  --n_samples 100 \
  --batch_size 32 \
  --patience 5 \
  --itr 1

# ETTh1 96->192
python -u run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --n_heads 8 \
  --diffusion_steps 1000 \
  --beta_schedule cosine \
  --cond_dim 256 \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --stage1_lr 1e-4 \
  --stage2_lr 1e-5 \
  --n_samples 100 \
  --batch_size 32 \
  --patience 5 \
  --itr 1

# ETTh1 96->336
python -u run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --n_heads 8 \
  --diffusion_steps 1000 \
  --beta_schedule cosine \
  --cond_dim 256 \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --stage1_lr 1e-4 \
  --stage2_lr 1e-5 \
  --n_samples 100 \
  --batch_size 32 \
  --patience 5 \
  --itr 1

# ETTh1 96->720
python -u run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --n_heads 8 \
  --diffusion_steps 1000 \
  --beta_schedule cosine \
  --cond_dim 256 \
  --stage1_epochs 30 \
  --stage2_epochs 20 \
  --stage1_lr 1e-4 \
  --stage2_lr 1e-5 \
  --n_samples 100 \
  --batch_size 32 \
  --patience 5 \
  --itr 1
