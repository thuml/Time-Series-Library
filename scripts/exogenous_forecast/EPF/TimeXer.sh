export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer
des='Timexer-MS'
patch_len=24


python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path NP.csv \
  --model_id NP_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len $patch_len \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path PJM.csv \
  --model_id PJM_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len $patch_len \
  --d_model 512 \
  --batch_size 16 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path BE.csv \
  --model_id BE_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len $patch_len \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path FR.csv \
  --model_id FR_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len $patch_len \
  --batch_size 16 \
  --d_model 512 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path DE.csv \
  --model_id DE_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --e_layers 1 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len $patch_len \
  --batch_size 4 \
  --d_model 512 \
  --itr 1
