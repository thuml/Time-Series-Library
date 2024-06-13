# source /d/anaconda3/etc/profile.d/conda.sh && conda activate autoformer
export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path /kaggle/input/anomaly-data/dataset/SWaT \
  --model_id SWAT \
  --model AutoKANFormer \
  --data SWAT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 10