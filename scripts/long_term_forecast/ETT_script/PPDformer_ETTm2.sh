export CUDA_VISIBLE_DEVICES=2


model_name=PPDformer
patchH=2
# patchW=2
strideH=1
# strideW=4
seq_len=96


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --patchH $patchH \
  --top_k 5 \
  --normal 0 \
  --dropout 0.1 \
  --attention 1 \
  --patchW 2\
  --strideH $strideH\
  --strideW 2\
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --patchH $patchH \
  --top_k 5 \
  --attention 1 \
  --patchW 4\
  --normal 1 \
  --dropout 0.0 \
  --strideH $strideH\
  --strideW 4\
  --n_heads 8 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --patchH $patchH \
  --top_k 5 \
  --attention 1 \
  --patchW 16\
  --normal 1 \
  --dropout 0.0 \
  --strideH $strideH\
  --strideW 8\
  --n_heads 4 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --patchH $patchH \
  --top_k 5 \
  --attention 1 \
  --patchW 4\
  --normal 1 \
  --dropout 0.0 \
  --strideH $strideH\
  --strideW 4\
  --n_heads 16 \
  --itr 1