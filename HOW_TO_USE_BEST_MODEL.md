# 如何使用最佳模型配置

**最佳MSE**: 0.6452 (Phase 3优化后)

---

## 1. 使用已训练模型进行预测

### 测试模式（使用现有checkpoint）

```bash
conda activate tslib

python run.py \
  --task_name diffusion_forecast \
  --is_training 0 \
  --model_id ETTh1_96 \
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
  --use_amp \
  --des 'Test'
```

**说明**:
- `--is_training 0`: 测试模式，不训练
- 自动加载checkpoint: `checkpoints/diffusion_forecast_ETTh1_96_iTransformerDiffusionDirect_.../checkpoint.pth`
- 输出: `results/` 目录下的预测结果

---

## 2. 从头训练（复现最佳结果）

### 训练脚本

```bash
conda activate tslib

python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model_id ETTh1_96 \
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
  --train_epochs 30 \
  --patience 3 \
  --learning_rate 0.0001 \
  --loss_lambda 0.5 \
  --n_samples 100 \
  --use_ddim \
  --ddim_steps 50 \
  --chunk_size 10 \
  --use_amp \
  --des 'Reproduce_Best'
```

**预期结果**:
- MSE: ~0.645
- 训练时间: ~15-20分钟 (RTX GPU)
- Checkpoint保存在: `checkpoints/...Reproduce_Best_0/`

---

## 3. 关键参数说明

### 必须启用的参数（保证最佳MSE）

```python
# 在模型的predict方法中，MoM默认已启用
use_mom = True   # ⭐ 关键！降低MSE 8.6%
mom_k = 10       # MoM分组数，推荐值
```

**注意**: 当前代码中MoM已默认启用，无需额外配置。

### 采样参数（平衡精度与速度）

| 参数 | 值 | 说明 |
|------|---|------|
| `n_samples` | 100 | 概率预测样本数（越多越好，但更慢） |
| `use_ddim` | True | 使用DDIM加速采样（推荐） |
| `ddim_steps` | 50 | DDIM步数（50是精度和速度的平衡点） |
| `chunk_size` | 10 | 分块采样大小（控制显存） |

### 训练参数（复现最佳结果）

| 参数 | 值 | 说明 |
|------|---|------|
| `train_epochs` | 30 | 训练轮数（已验证最优） |
| `patience` | 3 | 早停耐心值 |
| `loss_lambda` | 0.5 | MSE与扩散损失权重（平衡点预测和概率质量） |
| `learning_rate` | 0.0001 | 学习率 |
| `use_amp` | True | 混合精度（节省30-50%显存） |

---

## 4. 在新数据集上使用

### 修改数据集参数

```bash
python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model iTransformerDiffusionDirect \
  --data <YOUR_DATASET> \           # 修改：数据集名称
  --root_path <YOUR_PATH> \          # 修改：数据路径
  --data_path <YOUR_FILE.csv> \      # 修改：数据文件
  --enc_in <N_VARIATES> \            # 修改：输入变量数
  --dec_in <N_VARIATES> \            # 修改：解码器输入变量数
  --c_out <N_VARIATES> \             # 修改：输出变量数
  --seq_len 96 \                     # 可选：输入序列长度
  --pred_len 96 \                    # 可选：预测长度
  # ... 其他参数保持不变
```

**示例：使用Weather数据集**

```bash
python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model iTransformerDiffusionDirect \
  --data Weather \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  # ... 其他参数相同
```

---

## 5. 评估指标说明

### 点预测指标

- **MSE** (Mean Squared Error): 均方误差，越小越好
  - 最佳: 0.6452
- **MAE** (Mean Absolute Error): 平均绝对误差
  - 最佳: 0.5260
- **RMSE** (Root MSE): MSE的平方根
  - 最佳: 0.8033

### 概率预测指标

- **CRPS** (Continuous Ranked Probability Score): 连续排序概率分数
  - 最佳: 0.4889
  - 综合评估预测分布质量

- **校准度 (Calibration)**:
  - 50%覆盖率: 0.4603 (理想值0.5)
  - 90%覆盖率: 0.8569 (理想值0.9)
  - 衡量预测区间的准确性

- **锐度 (Sharpness)**: 0.6150
  - 预测区间的平均宽度
  - 越小说明模型越自信（但要配合校准度评估）

---

## 6. 常见问题

### Q1: 为什么概率模型的MSE比确定性模型（如PatchTST）高？

**A**: 这是**正常且预期的**：
- 确定性模型（PatchTST）: MSE=0.377，但**无法提供不确定性信息**
- 概率模型（iTransformerDiffusion）: MSE=0.645，但**提供完整的预测分布**
- 概率模型的价值在于**不确定性量化**，而不是最小化MSE

### Q2: 如何选择n_samples和ddim_steps？

**A**: 根据精度和速度需求权衡：

| 配置 | n_samples | ddim_steps | 速度 | 精度 |
|------|----------|-----------|------|------|
| **快速** | 50 | 25 | ⚡⚡⚡ | ⭐⭐ |
| **推荐** | 100 | 50 | ⚡⚡ | ⭐⭐⭐ |
| **高精度** | 200 | 100 | ⚡ | ⭐⭐⭐⭐ |

### Q3: 显存不足怎么办？

**A**: 调整以下参数：
```bash
--batch_size 16          # 减小batch size (默认32)
--chunk_size 5           # 减小chunk size (默认10)
--n_samples 50           # 减少采样数 (默认100)
--use_amp                # 启用混合精度 (节省30-50%显存)
```

### Q4: MoM优化是否总是启用？

**A**: 是的，当前代码中`use_mom=True`是默认配置，已经集成在模型的`predict`方法中。无需额外配置即可享受8.6%的MSE改善。

---

## 7. Checkpoint文件说明

### 最佳Checkpoint路径

```
checkpoints/diffusion_forecast_ETTh1_96_iTransformerDiffusionDirect_ETTh1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Fixed_MSE_0/checkpoint.pth
```

**大小**: ~39MB

**包含内容**:
- 模型权重（iTransformer backbone + 扩散网络）
- 归一化统计量（instance normalization）
- 优化器状态（如需继续训练）

---

## 8. 快速上手示例

### 最简单的使用方式

```bash
# 1. 激活环境
conda activate tslib

# 2. 进入项目目录
cd ~/projects/Time-Series-Library

# 3. 运行测试（使用最佳checkpoint）
python run.py \
  --task_name diffusion_forecast \
  --is_training 0 \
  --model_id ETTh1_96 \
  --model iTransformerDiffusionDirect \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 128 --d_ff 128 --e_layers 2 --n_heads 8 \
  --diffusion_steps 1000 --beta_schedule cosine \
  --n_samples 100 --use_ddim --ddim_steps 50 \
  --use_amp

# 4. 查看结果
tail result_diffusion_forecast.txt
```

**预期输出**:
```
Point: mse:0.645241, mae:0.526038, rmse:0.803269
Prob: crps:0.488904, calib_50:0.4603, calib_90:0.8569
```

---

*文档更新时间: 2026-01-23*
*对应模型: iTransformerDiffusionDirect (Phase 3 最佳配置)*
