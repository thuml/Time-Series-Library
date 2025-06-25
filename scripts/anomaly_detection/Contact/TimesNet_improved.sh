export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset \
  --model_id Contact_improved \
  --model TimesNet \
  --data Contact \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 27 \
  --c_out 27 \
  --top_k 5 \
  --anomaly_ratio 5.2 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 5 