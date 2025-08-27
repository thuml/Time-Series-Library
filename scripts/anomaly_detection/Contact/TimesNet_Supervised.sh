#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

echo "🚀 启动基于TimesNet的监督异常检测训练..."
echo "📊 数据: 能源Contact数据集 (标注版本)"
echo "🏗️ 模型: TimesNet + 逐点分类头"
echo "🎯 任务: 逐点异常检测 (每分钟数据点分类)"



python3 -u run.py \
  --task_name supervised_anomaly_detection \
  --is_training 1 \
  --root_path ./dataset \
  --model_id SupervisedContact \
  --model TimesNet \
  --data SupervisedContact \
  --features M \
  --seq_len 40 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 27 \
  --c_out 1 \
  --top_k 3 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 3 \
  --patience 10 \
  --num_workers 4 \
  --des 'TimesNet_Supervised_Contact_Anomaly_Detection' \
  --itr 1 