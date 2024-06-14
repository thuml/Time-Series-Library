#!/bin/bash

# source /d/anaconda3/etc/profile.d/conda.sh && conda activate autoformer
export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /kaggle/input/anomaly-data/dataset/VincomRoyal_processed1 \
  --model_id PSM \
  --model AutoKANFormer \
  --data PSM \
  --features M \
  --seq_len 64 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 1 \
  --c_out 1 \
  --anomaly_ratio 10 \
  --batch_size 64 \
  --train_epochs 15 \
  --patience 10


# python -u run.py \
#   --task_name anomaly_detection \
#   --is_training 1 \
#   --root_path ./dataset/VincomRoyal_processed1 \
#   --model_id PSM \
#   --model AutoKANFormer \
#   --data PSM \
#   --features M \
#   --seq_len 96 \
#   --pred_len 0 \
#   --d_model 128 \
#   --d_ff 128 \
#   --e_layers 3 \
#   --enc_in 1 \
#   --c_out 1 \
#   --anomaly_ratio 10 \
#   --batch_size 64 \
#   --train_epochs 15