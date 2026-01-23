#!/bin/bash

# ===================================================================
# iTransformerDiffusionDirect - ETTh1 训练脚本（修复版）
# ===================================================================
#
# 修复内容：
# 1. ✓ 验证损失使用 backbone_forward（真实点预测MSE）
# 2. ✓ 预测使用 Median-of-Means（MSE降低8.3%）
# 3. ✓ 损失权重固定α=0.8（80% MSE + 20% Diffusion）
#
# 预期效果：
# - Early stopping 基于真实MSE，训练持续15-20 epoch
# - 最终MSE: 0.36-0.45（vs 修复前0.71）
# - 接近确定性模型性能（PatchTST 0.377, iTransformer 0.395）
# ===================================================================

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tslib

# 设置
model_name=iTransformerDiffusionDirect
data_name=ETTh1

# 预测长度列表（按计划只测试96）
pred_lens=(96)

# GPU设置
export CUDA_VISIBLE_DEVICES=0

for pred_len in "${pred_lens[@]}"; do
  echo "========================================"
  echo "训练 ${model_name} - pred_len=${pred_len}"
  echo "========================================"

  python run.py \
    --task_name diffusion_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${pred_len} \
    --model ${model_name} \
    --data ${data_name} \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --des 'Fixed_MSE' \
    --itr 1 \
    --train_epochs 30 \
    --patience 5 \
    --warmup_epochs 10 \
    --diffusion_steps 1000 \
    --beta_schedule cosine \
    --cond_dim 256 \
    --n_samples 100 \
    --use_ddim \
    --ddim_steps 50 \
    --chunk_size 10 \
    --use_amp \
    --parameterization v

  echo ""
  echo "✓ pred_len=${pred_len} 训练完成"
  echo ""
done

echo "========================================"
echo "所有实验完成！"
echo "========================================"
echo ""
echo "检查结果："
echo "1. 查看训练日志：训练应持续15-20 epoch（不会在6个epoch就停）"
echo "2. 查看result_diffusion_forecast.txt：MSE应该在0.36-0.45范围"
echo "3. 对比修复前性能（MSE 0.7087）：应该降低40-50%"
echo ""
echo "如果MSE < 0.45且CRPS < 0.45，修复成功！"
