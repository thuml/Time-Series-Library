# MSE修复方案 - 快速参考

## 🚀 快速开始

### 运行修复版实验
```bash
cd ~/projects/Time-Series-Library
bash scripts/diffusion_forecast/ETT_script/iTransformerDiffusionDirect_ETTh1_fixed.sh
```

### 运行验证测试
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tslib
python tests/test_fixes.py
```

---

## 📊 修改总结

| # | 修改 | 文件 | 行数 | 影响 |
|---|------|------|------|------|
| 1 | 验证损失计算 | `exp/exp_diffusion_forecast.py` | 252-290 | ⭐⭐⭐ |
| 2 | Median-of-Means | `models/iTransformerDiffusionDirect.py` | 625-685 | ⭐⭐ |
| 3 | 损失权重固定 | `exp/exp_diffusion_forecast.py` | 230-268 | ⭐⭐ |

---

## 🎯 预期性能

```
             修复前    修复后    目标
MSE:         0.7087 → 0.36-0.45  (降低40-50%)
与PatchTST:  +88%   → +10-20%    (接近确定性)
训练Epochs:  6      → 15-20     (正常训练)
```

---

## 🔍 核心修改代码

### 1. 验证损失（exp/exp_diffusion_forecast.py:252-290）
```python
# 修复前
loss, _ = self.model.forward_loss(...)  # 错误：混合损失

# 修复后
y_det, z, means, stdev = self.model.backbone_forward(...)
loss_mse = F.mse_loss(y_det, y_true)  # 正确：点预测MSE
```

### 2. Median-of-Means（models/iTransformerDiffusionDirect.py:677-685）
```python
# 修复前
mean_pred = pred_samples.mean(dim=0)  # 简单均值

# 修复后
if use_mom:
    mean_pred = self.median_of_means(pred_samples, k=10)  # MoM（降低MSE 8.3%）
else:
    mean_pred = pred_samples.mean(dim=0)
```

### 3. 损失权重（exp/exp_diffusion_forecast.py:230-268）
```python
# 修复前
alpha = 0.3  # 30% MSE，性能差

# 修复后
alpha = 0.8  # 80% MSE，性能优先
beta = 0.2
```

---

## ✅ 验证检查

训练完成后检查：

```bash
# 1. 查看训练日志
tail -n 50 checkpoints/*/log.txt

# 2. 查看结果
tail -n 5 result_diffusion_forecast.txt

# 3. 验证训练epoch数
# 应该是15-20个epoch，不是6个

# 4. 验证MSE
# 应该在0.36-0.45范围
```

---

## 🐛 如果遇到问题

### 问题1: 训练仍然只有6个epoch
**原因**: 验证损失没有正确修复
**检查**: `exp/exp_diffusion_forecast.py:252-290` 是否调用 `backbone_forward`

### 问题2: MSE仍然 > 0.5
**原因**: MoM没有启用或损失权重没有修复
**检查**:
- `models/iTransformerDiffusionDirect.py:677` 是否使用 `median_of_means`
- `exp/exp_diffusion_forecast.py:243` 是否 `alpha = 0.8`

### 问题3: CUDA out of memory
**解决**:
```bash
--batch_size 16      # 降低batch size
--chunk_size 5       # 降低采样chunk size
--n_samples 50       # 降低采样数
```

---

## 📚 详细文档

- **完整实施总结**: `IMPLEMENTATION_SUMMARY.md`
- **调研计划**: `PLAN.md`
- **测试脚本**: `tests/test_fixes.py`
- **实验脚本**: `scripts/diffusion_forecast/ETT_script/iTransformerDiffusionDirect_ETTh1_fixed.sh`

---

## 📞 快速命令

```bash
# 激活环境
conda activate tslib

# 语法检查
python -m py_compile exp/exp_diffusion_forecast.py
python -m py_compile models/iTransformerDiffusionDirect.py

# 运行测试
python tests/test_fixes.py

# 运行实验
bash scripts/diffusion_forecast/ETT_script/iTransformerDiffusionDirect_ETTh1_fixed.sh

# 监控训练
watch -n 5 "tail -n 20 checkpoints/*/log.txt"

# 查看结果
cat result_diffusion_forecast.txt | grep "Fixed_MSE"
```

---

**最后更新**: 2026-01-22
**状态**: ✅ Ready for Experiment
