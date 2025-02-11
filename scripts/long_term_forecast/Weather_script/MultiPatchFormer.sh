export CUDA_VISIBLE_DEVICES=0

model_name=MultiPatchFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1