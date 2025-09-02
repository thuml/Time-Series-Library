export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWAT \
  --model KANAD \
  --data SWAT \
  --features M \
  --seq_len 80 \
  --d_model 1 \
  --enc_in 51 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --learning_rate 0.01 \
  --batch_size 128 \
  --num_workers 4 \
  --patience 5 \
  --train_epochs 100