export CUDA_VISIBLE_DEVICES=2

model_name=TiDE

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 8 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 512 \
  --learning_rate 0.1 \
  --patience 5 \
  --train_epochs 10 \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 8 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 512 \
  --learning_rate 0.1 \
  --patience 5 \
  --train_epochs 10 \

 


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 8 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 512 \
  --learning_rate 0.1 \
  --patience 5 \
  --train_epochs 10 \




python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 8 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 512 \
  --learning_rate 0.1 \
  --patience 5 \
  --train_epochs 10 \

