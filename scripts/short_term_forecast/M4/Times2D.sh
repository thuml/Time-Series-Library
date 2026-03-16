export CUDA_VISIBLE_DEVICES=0

model_name=Times2D

# =========================
# Yearly
# =========================
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns Yearly \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 512 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --top_k 5 \
  --loss 'SMAPE' \
  --train_epochs 100 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1

# =========================
# Quarterly
# =========================
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns Quarterly \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --top_k 5 \
  --loss 'SMAPE' \
  --train_epochs 100 \
  --patience 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1

# =========================
# Monthly
# =========================
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns Monthly \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_ff 256 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --top_k 5 \
  --loss 'SMAPE' \
  --train_epochs 100 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1


# =========================
# Weekly
# =========================
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns Weekly \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 32 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --top_k 5 \
  --loss 'SMAPE' \
  --train_epochs 100 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1

# =========================
# Daily
# =========================
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns Daily \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 16 \
  --d_ff 16 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --top_k 5 \
  --loss 'SMAPE' \
  --train_epochs 100 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1

# =========================
# Hourly
# =========================
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns Hourly \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 32 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_length 48 32 16 6 3 \
  --top_k 5 \
  --loss 'SMAPE' \
  --train_epochs 100 \
  --patience 10 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1