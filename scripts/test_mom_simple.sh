#!/bin/bash
# 简化的MoM测试脚本
# 使用不同的MoM配置测试同一个模型

CHECKPOINT_DIR="diffusion_forecast_ETTh1_96_iTransformerDiffusionDirect_ETTh1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Fixed_MSE_0"

echo "=========================================="
echo "MoM 优化测试"
echo "=========================================="
echo ""

# 基础参数
COMMON_ARGS="--task_name diffusion_forecast \
  --is_training 0 \
  --model iTransformerDiffusionDirect \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 2 \
  --n_heads 8 \
  --diffusion_steps 1000 \
  --beta_schedule cosine \
  --parameterization x0 \
  --n_samples 100 \
  --use_ddim \
  --ddim_steps 50 \
  --chunk_size 10 \
  --use_amp"

# 注意：当前模型代码中use_mom默认为True
# 我们需要修改代码或者通过其他方式测试

echo "由于无法直接通过命令行参数控制use_mom，"
echo "我们需要通过修改模型代码或创建Python测试脚本。"
echo ""
echo "建议：创建一个简单的Python脚本来直接调用模型的predict方法"
