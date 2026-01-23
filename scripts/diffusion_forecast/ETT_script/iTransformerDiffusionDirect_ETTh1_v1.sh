#!/bin/bash

# 阶段 6：完整训练验证 (10 epoch)
# iTransformerDiffusionDirect with v-prediction on ETTh1
#
# 优化内容：
# 1. v-prediction 参数化（更稳定）
# 2. 端到端联合训练（梯度连通）
# 3. 课程学习权重调度
# 4. AMP 混合精度（省显存）

python -u run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_v1 \
  --model iTransformerDiffusionDirect \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Phase1_Final' \
  --itr 1 \
  --parameterization v \
  --training_mode end_to_end \
  --train_epochs 10 \
  --warmup_epochs 3 \
  --learning_rate 1e-4 \
  --batch_size 32 \
  --diffusion_steps 100 \
  --n_samples 10 \
  --use_ddim \
  --ddim_steps 10 \
  --use_amp

echo "阶段 6 完整训练验证完成！"
