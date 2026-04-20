#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=Times2D
root_path=./dataset/ETT-small/
data_path=ETTh2.csv
data_name=ETTh2
seq_len=720


# =======================
# pred_len = 96
# =======================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTh2_720_96 \
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
  --kernel_list 5 7 11 15 \
  --patch_length 48 32 16 6 3 \
  --stride 48 32 16 6 3 \
  --dropout 0.6 \
  --fc_dropout 0.25 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 128 \
  --learning_rate 0.0001


# =======================
# pred_len = 192
# =======================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTh2_720_192 \
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
  --d_model 32 \
  --d_ff 128 \
  --kernel_list 5 7 11 15 \
  --patch_length 48 32 16 6 3 1 \
  --stride 48 32 16 6 3 \
  --dropout 0.6 \
  --fc_dropout 0.25 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 128 \
  --learning_rate 0.0001


# =======================
# pred_len = 336
# =======================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTh2_720_336 \
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
  --kernel_list 5 7 11 15 \
  --patch_length 48 32 16 6 3 \
  --stride 48 32 16 6 3 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 128 \
  --learning_rate 0.0001


# =======================
# pred_len = 720
# =======================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id ETTh2_720_720 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 96 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 32 \
  --kernel_list 5 7 11 15 \
  --patch_length 48 32 16 6 3 \
  --stride 48 32 16 6 3 \
  --dropout 0.6 \
  --fc_dropout 0.25 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 128 \
  --learning_rate 0.0001
