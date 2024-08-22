
seq_len=336
model_name=PatchMixer

root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

for pred_len in 96 192 336 720
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1\
      --d_model 256 \
      --dropout 0.2\
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 10 \
      --itr 1 --batch_size 1024 --learning_rate 0.0001
done