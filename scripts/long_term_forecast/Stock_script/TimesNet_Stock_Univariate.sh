export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/stock_market \
  --data_path ./dataset/stock_market \
  --model_id stock_forecast_univariate \
  --model $model_name \
  --data stock_market \
  --features S \
  --target Close \
  --freq d \
  --seq_len 32 \
  --label_len 16 \
  --pred_len 16 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 2 \
  --train_epochs 3 \
  --learning_rate 0.0001 \
  --num_workers 2 \
  --e_layers 2 \
  --d_layers 1

# Prediction script
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/stock_market \
  --data_path ./dataset/stock_market \
  --model_id stock_forecast_univariate \
  --model $model_name \
  --data stock_market \
  --features S \
  --target Close \
  --freq d \
  --seq_len 32 \
  --label_len 16 \
  --pred_len 16 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 2 