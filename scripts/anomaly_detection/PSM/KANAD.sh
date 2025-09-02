export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model KANAD \
  --data PSM \
  --features M \
  --seq_len 64 \
  --d_model 6 \
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --learning_rate 0.01 \
  --batch_size 128 \
  --num_workers 4 \
  --patience 5 \
  --train_epochs 100