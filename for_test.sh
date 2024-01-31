export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path tpc_RF.csv \
  --model_id wind_12_12 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 12 \
  --label_len 5 \
  --pred_len 12 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 64\
  --d_ff 64\
  --itr 1 \
