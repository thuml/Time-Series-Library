# Phase 2 优化实施准备就绪 ✓

## 📋 完成的工作

### 1. 训练脚本创建 ✓

#### 基线训练脚本
- **文件**: `scripts/diffusion_forecast/ETT_script/iTransformerDiffusionDirect_ETTh1_baseline.sh`
- **配置**:
  - d_model=128, e_layers=2 (增强容量)
  - diffusion_steps=500 (更充分扩散)
  - train_epochs=30, warmup_epochs=6
  - n_samples=50, ddim_steps=25
- **预期**: MSE 0.50-0.60, CRPS 0.40-0.45

#### 对比模型训练脚本
1. **iTransformer**: `scripts/long_term_forecast/ETT_script/iTransformer_ETTh1.sh`
   - Backbone 基线，d_model=128, e_layers=2

2. **PatchTST**: `scripts/long_term_forecast/ETT_script/PatchTST_ETTh1_baseline.sh`
   - Patch-based 架构对比

3. **TimesNet**: `scripts/long_term_forecast/ETT_script/TimesNet_ETTh1_baseline.sh`
   - Multi-period 架构对比

### 2. FR2 优化准备 ✓

#### 代码实现
- **模块**: `layers/Diffusion_layers.py` 中的 `FrequencyAwareResidual` 类
- **功能**: 频域感知残差连接，增强频域表达能力
- **测试**: 全部通过（`tests/validate_fr2.py`）

#### 集成文档
- **文件**: `docs/FR2_INTEGRATION_GUIDE.md`
- **内容**: 详细的集成步骤、配置参数、验证测试、消融实验指南

### 3. 启动脚本 ✓

#### 一键启动脚本
- **文件**: `scripts/phase2_launch_all.sh`
- **功能**: 并行启动所有训练任务（基线 + 3 个对比模型）
- **特性**: 自动创建日志目录，后台运行，提供监控命令

---

## 🚀 启动训练

### 方式 1: 一键启动所有训练（推荐）

```bash
cd ~/projects/Time-Series-Library
bash scripts/phase2_launch_all.sh
```

**预计时间**: 4-6 小时（并行执行）

**监控进度**:
```bash
# 查看所有日志
tail -f logs/phase2/*.log

# 查看基线训练日志
tail -f logs/phase2/baseline.log

# 查看特定模型
tail -f logs/phase2/iTransformer.log

# 查看进程状态
ps aux | grep python
```

### 方式 2: 单独启动训练

```bash
# 只启动基线训练
bash scripts/diffusion_forecast/ETT_script/iTransformerDiffusionDirect_ETTh1_baseline.sh

# 只启动对比模型
bash scripts/long_term_forecast/ETT_script/iTransformer_ETTh1.sh
```

---

## 📊 训练完成后的分析

### 1. 收集结果

训练完成后，结果保存在：
```
checkpoints/
├── diffusion_forecast_ETTh1_96_96_baseline_iTransformerDiffusionDirect_*/
├── long_term_forecast_ETTh1_96_96_baseline_iTransformer_*/
├── long_term_forecast_ETTh1_96_96_baseline_PatchTST_*/
└── long_term_forecast_ETTh1_96_96_baseline_TimesNet_*/
```

每个目录包含：
- `result.txt` - 测试集指标
- `metrics.npy` - 详细指标数据
- `checkpoint.pth` - 最佳模型

### 2. 对比分析

创建对比表格（示例）：

| 模型 | MSE | MAE | CRPS | 校准50% | 校准90% | 训练时间 |
|------|-----|-----|------|---------|---------|---------|
| iTransformer | ? | ? | N/A | N/A | N/A | ? |
| PatchTST | ? | ? | N/A | N/A | N/A | ? |
| TimesNet | ? | ? | N/A | N/A | N/A | ? |
| **Baseline (ours)** | ? | ? | ? | ? | ? | ? |

### 3. 决策流程

```
基线训练完成
    │
    ├─ MSE < 0.60 且 CRPS < 0.45？
    │   ├─ 是 → Phase 2 成功，继续 Phase 3 优化
    │   └─ 否 ↓
    │
    ├─ 频域损失明显偏高？
    │   ├─ 是 → 实施 FR2 优化
    │   └─ 否 ↓
    │
    ├─ 校准度差（50% < 0.40 或 90% < 0.80）？
    │   ├─ 是 → 实施温度缩放
    │   └─ 否 ↓
    │
    └─ 增大模型容量
        ├─ diffusion_steps: 500 → 1000
        ├─ train_epochs: 30 → 50
        └─ d_model: 128 → 256
```

---

## 🔧 FR2 优化集成（按需）

**仅在基线结果显示需要时才集成 FR2**

### 何时集成
- 频域损失 `loss_freq` 明显高于对比模型
- 周期性预测不准确
- 整体 CRPS 距离目标较远

### 集成步骤
详见：`docs/FR2_INTEGRATION_GUIDE.md`

### 验证 FR2
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tslib
python tests/validate_fr2.py
```

---

## 📁 文件清单

### 新建文件
```
scripts/
├── diffusion_forecast/ETT_script/
│   └── iTransformerDiffusionDirect_ETTh1_baseline.sh
├── long_term_forecast/ETT_script/
│   ├── iTransformer_ETTh1.sh
│   ├── PatchTST_ETTh1_baseline.sh
│   └── TimesNet_ETTh1_baseline.sh
└── phase2_launch_all.sh

layers/
└── Diffusion_layers.py  (新增 FrequencyAwareResidual 类)

tests/
├── test_fr2.py
└── validate_fr2.py

docs/
├── FR2_INTEGRATION_GUIDE.md
└── PHASE2_READY.md  (本文件)
```

### 修改文件
- `layers/Diffusion_layers.py` - 新增 FR2 模块

---

## ⏱️ 时间估算

| 任务 | 预计时间 | 备注 |
|------|---------|------|
| 基线训练 | 2-3小时 | 30 epochs |
| iTransformer | 1.5-2小时 | 30 epochs |
| PatchTST | 1.5-2小时 | 30 epochs |
| TimesNet | 1.5-2小时 | 30 epochs |
| **总计（并行）** | **4-6小时** | 同时运行 |
| **总计（串行）** | **7-9小时** | 依次运行 |

---

## 🎯 成功标准

### Phase 2.1 完成标准
- ✅ 基线模型 MSE < 0.60
- ✅ 完成与 3 个确定性模型的对比
- ✅ 识别出 1-2 个关键瓶颈
- ✅ （可选）实施并验证 1 个优化的效果
- ✅ 输出详细的对比分析报告

### 长期目标（Phase 3+）
- 🎯 MSE < 0.45（接近历史最佳）
- 🎯 CRPS < 0.40（优于历史）
- 🎯 校准度：50%=0.45-0.55, 90%=0.85-0.95
- 🎯 训练稳定性：全程无 NaN

---

## 🆘 故障排查

### 训练卡住
```bash
# 查看进程状态
ps aux | grep python

# 查看 GPU 使用
nvidia-smi

# 查看日志
tail -f logs/phase2/*.log
```

### 内存不足
修改脚本中的 `--batch_size`:
```bash
--batch_size 16  # 从 32 降低到 16
```

### CUDA 错误
确认 GPU 可用:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tslib
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📞 下一步

1. **立即执行**: `bash scripts/phase2_launch_all.sh`
2. **等待完成**: 4-6 小时
3. **分析结果**: 对比所有模型指标
4. **决策优化**: 根据分析结果选择 Phase 2.3 的优化方向

---

**创建时间**: 2026-01-22
**状态**: ✅ 准备就绪，可以启动训练
**预计完成**: 2026-01-22 晚上（4-6 小时后）
