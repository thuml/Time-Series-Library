#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=Times2D

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 36 \
  --pred_len 24 \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 64 \
  --d_ff 64 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --patience 10


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 36 \
  --pred_len 36 \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 64 \
  --d_ff 64 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --patience 10


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 36 \
  --pred_len 48 \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 64 \
  --d_ff 64 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --patience 10


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 36 \
  --pred_len 60 \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 64 \
  --d_ff 64 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --patience 10
