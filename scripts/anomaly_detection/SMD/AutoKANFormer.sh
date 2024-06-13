source /d/anaconda3/etc/profile.d/conda.sh && conda activate autoformer
export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /kaggle/input/anomaly-data/dataset/SMD \
  --model_id SMD \
  --model AutoKANFormer \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 0.5 \
  --batch_size 64 \
  --train_epochs 10