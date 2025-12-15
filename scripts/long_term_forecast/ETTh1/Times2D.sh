#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=Times2D
root_path=./dataset/ETT-small/
data_path=ETTh1.csv
data_name=ETTh1
seq_len=720

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTh1_720_96 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 64 \
  --patch_length 48 32 16 6 3 \
  --dropout 0.5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 128 \
  --learning_rate 0.0001


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTh1_720_192 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 64 \
  --patch_length 48 32 16 6 3 \
  --dropout 0.5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 128 \
  --learning_rate 0.0001


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTh1_720_336 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 64 \
  --patch_length 48 32 16 6 3 \
  --dropout 0.5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 128 \
  --learning_rate 0.0001


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTh1_720_720 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 64 \
  --d_ff 64 \
  --patch_length 48 32 16 6 3 \
  --dropout 0.5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 128 \
  --learning_rate 0.0001
