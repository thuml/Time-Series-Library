export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset \
  --model_id Contact \
  --model TimesNet \
  --data Contact \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 27 \
  --c_out 27 \
  --top_k 3 \
  --anomaly_ratio 5.2 \
  --batch_size 128 \
  --train_epochs 3 