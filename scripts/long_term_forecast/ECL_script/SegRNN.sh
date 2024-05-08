export CUDA_VISIBLE_DEVICES=0

model_name=SegRNN

seq_len=96
for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len 24 \
  --enc_in 321 \
  --d_model 512 \
  --dropout 0 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1
done

