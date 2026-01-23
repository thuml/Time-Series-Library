#!/bin/bash

# Phase 2.1: 基线训练 (保守配置)
# iTransformerDiffusionDirect Baseline Training on ETTh1
#
# 配置调整 (相比 Phase 1):
# - d_model: 64 → 128 (匹配 iTransformer)
# - e_layers: 1 → 2 (增加表达能力)
# - diffusion_steps: 100 → 500 (更充分的扩散)
# - train_epochs: 10 → 30 (充分训练)
# - warmup_epochs: 3 → 6 (更平滑的课程学习)
# - patience: 3 → 5 (减少过早停止)
# - n_samples: 10 → 50 (更准确的概率估计)
# - ddim_steps: 10 → 25 (提高采样质量)
#
# 预期目标:
# - MSE: 0.50-0.60
# - CRPS: 0.40-0.45
# - 校准 50%: 0.40-0.60
# - 校准 90%: 0.80-0.95

python -u run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_baseline \
  --model iTransformerDiffusionDirect \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Phase2_Baseline_Conservative' \
  --itr 1 \
  --parameterization v \
  --training_mode end_to_end \
  --train_epochs 30 \
  --warmup_epochs 6 \
  --learning_rate 1e-4 \
  --batch_size 32 \
  --patience 5 \
  --diffusion_steps 500 \
  --n_samples 50 \
  --use_ddim \
  --ddim_steps 25 \
  --use_amp

echo "Phase 2.1 基线训练完成！"
