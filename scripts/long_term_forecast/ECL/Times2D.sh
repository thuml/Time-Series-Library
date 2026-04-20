#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=Times2D

#############################################
# pred_len = 96
#############################################
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 96 \
  --enc_in 321 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 64 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --kernel_list 5 7 11 15 \
  --patch_length 48 32 16 6 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001


#############################################
# pred_len = 192
#############################################
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 192 \
  --enc_in 321 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 64 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --kernel_list 5 7 11 15 \
  --patch_length 48 32 16 6 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001


#############################################
# pred_len = 336
#############################################
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 336 \
  --enc_in 321 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 64 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --kernel_list 5 7 11 15 \
  --patch_length 48 32 16 6 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001


#############################################
# pred_len = 720
#############################################
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 720 \
  --enc_in 321 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 64 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --kernel_list 5 7 11 15 \
  --patch_length 48 32 16 6 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001
