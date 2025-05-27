model_name=TimesNet

# seq_len: How many past points to use for prediction, default is 96, set it to 32 to run in my machine
# label_len: number of overlapping time steps used during training, default is 48, set it to 16 to run in my machine
# pred_len: How many future points to predict, default is 96, set it to 16 to run in my machine
# factor: set it to 3 comparing it with other scripts
# enc_in: the number of features in the input ( 6 + 1 for company code)
# dec_in: the number of features in the output ( 6 + 1 for company code)
# c_out: the number of features in the output ( 6 + 1 for company code)
# d_model: default value is 512, had to set it to 64 to run in my machine
# d_ff: the default value is 2048, had to set it to 128 to run in my machine
# train_epochs: the number of epochs to train the model, default is 10, had to set it to 3
# batch_size: the number of samples to use in each batch, default is 32, had to set it to 2, to run in my machine
# num_workers: the number of workers to use for the model, default is 10, had to set it to 2, to run in my machine
# freq: default is h, set it to do as I was working with daily stock data

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
  --seq_len 32 \
  --label_len 16 \
  --pred_len 16 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3 \
  --batch_size 2 \
  --num_workers 2

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
  --seq_len 32 \
  --label_len 16 \
  --pred_len 16 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 2 