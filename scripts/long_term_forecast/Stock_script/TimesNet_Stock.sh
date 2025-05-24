export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/stock_market \
  --data_path ./dataset/stock_market \
  --model_id stock_forecast \
  --model $model_name \
  --data stock_market \
  --features MS \
  --target Close \
  --freq d \
  --seq_len 64 \
  --label_len 32 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 256 \
  --d_ff 1024 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 16 \
  --patience 3 \
  --learning_rate 0.0001 \
  --num_workers 4

# Prediction script
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/stock_market \
  --data_path ./dataset/stock_market \
  --model_id stock_forecast \
  --model $model_name \
  --data stock_market \
  --features MS \
  --target Close \
  --freq d \
  --seq_len 64 \
  --label_len 32 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 256 \
  --d_ff 1024 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 4 