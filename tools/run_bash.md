# Expirement

## Run the code in bash

### DyVolFusion

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/yfinance/ --data_path TSMC.csv --model_id TSMC_96_5 --model DyVolFusion --use_norm 1 --data custom --features MS --target Target_Vol --seq_len 96 --label_len 48 --pred_len 5 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 1 --d_model 64 --n_heads 4 --des Exp --itr 1 --train_epochs 20 --batch_size 32 --learning_rate 0.0001
```


