export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path formosa_wrf_short.csv \
  --model_id wind_12_12_short_wrf \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 12 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 19 \
  --dec_in 19 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 64\
  --d_ff 64\
  --itr 1 \
