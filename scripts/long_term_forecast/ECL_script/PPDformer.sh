export CUDA_VISIBLE_DEVICES=0


model_name=PPDformer
patchH=2
patchW=4
strideH=1
strideW=1
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1\
  --patchH $patchH \
  --top_k 5 \
  --attention 0 \
  --strideH $strideH\
  --strideW $strideW\
  --normal 1 \
  --dropout 0.1 \
  --patchW $patchW

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1\
  --normal 1 \
  --dropout 0.1 \
  --attention 0 \
  --strideH $strideH\
  --strideW $strideW\
  --patchH $patchH \
  --top_k 5 \
  --patchW $patchW


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1\
  --attention 0 \
  --normal 1 \
  --dropout 0.1 \
  --strideH $strideH\
  --strideW $strideW\
  --patchH $patchH \
  --top_k 5 \
  --patchW $patchW


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --normal 0 \
  --dropout 0.1 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1\
  --normal 1 \
  --dropout 0.1 \
  --attention 0 \
  --strideH $strideH\
  --strideW $strideW\
  --patchH $patchH \
  --top_k 5 \
  --patchW $patchW

