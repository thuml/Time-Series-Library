export CUDA_VISIBLE_DEVICES=1

model_name=Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/liyuante/llm4ts/ticker_csv_files/ \
  --data_path AAPL.csv \
  --seasonal_patterns 'Monthly' \
  --model_id usa_1 \
  --model $model_name \
  --data MY \
  --features M \
  --target RETX \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'