# 扩散模型MSE异常问题修复实施总结

**日期**: 2026-01-22
**状态**: ✅ 全部实施完成
**测试**: ✅ 所有单元测试通过

---

## 问题回顾

### 核心矛盾
iTransformerDiffusionDirect 的点预测MSE (0.7087) 比确定性模型（0.377-0.395）高出**近一倍**。

### 根本原因
通过深度调研（Phase 1）发现了三个关键问题：

1. **验证损失计算错误** ⭐⭐⭐
   - 位置：`exp/exp_diffusion_forecast.py:270`
   - 问题：使用 `forward_loss`（混合损失，数值600+）而非点预测MSE
   - 影响：Early stopping在Epoch 1就触发，训练过早结束

2. **点预测方法次优** ⭐⭐
   - 位置：`models/iTransformerDiffusionDirect.py:677`
   - 问题：使用简单均值，对异常值敏感
   - 改进：SimDiff的Median-of-Means方法（MSE降低8.3%）

3. **课程学习权重不合理** ⭐⭐
   - 位置：`exp/exp_diffusion_forecast.py:243-248`
   - 问题：后期α=0.3（70% Diffusion）导致点预测性能下降
   - 观察：Epoch 1 (α=1.0) 性能最佳，后续α降低反而MSE上升

---

## 实施方案

### ✅ Priority 1: 修复验证损失计算

**文件**: `exp/exp_diffusion_forecast.py`
**行数**: 252-290

**修改前**:
```python
def vali(self, vali_data, vali_loader, stage='warmup'):
    # 错误：使用 forward_loss（包含diffusion训练损失）
    loss, _ = self.model.forward_loss(batch_x, batch_x_mark, y_true, stage=stage)
    total_loss.append(loss.item())  # 数值600+，不合理
```

**修改后**:
```python
def vali(self, vali_data, vali_loader, stage='warmup'):
    # 正确：使用 backbone 确定性预测的MSE
    y_det, z, means, stdev = self.model.backbone_forward(batch_x, batch_x_mark)
    loss_mse = F.mse_loss(y_det, y_true)
    total_loss.append(loss_mse.item())  # 数值0.3-0.6，合理
```

**预期效果**:
- ✅ 验证损失数值合理（0.3-0.6）
- ✅ Early stopping基于真实点预测性能
- ✅ 训练持续15-20 epoch（vs 修复前6个）

---

### ✅ Priority 2: 实施SimDiff的Median-of-Means方法

**文件**: `models/iTransformerDiffusionDirect.py`
**行数**: 625-656（新增方法） + 677-685（修改predict）

**新增方法**:
```python
def median_of_means(self, samples, k=10):
    """
    Median-of-Means estimator (SimDiff方法)

    优势：
    1. MSE降低8.3%
    2. 对异常值更robust
    3. 保留temporal patterns（不过度平滑）

    方法：
    1. 将n_samples个样本分成k组
    2. 计算每组的均值
    3. 对k个均值取中位数
    """
    n_samples = samples.shape[0]
    group_size = n_samples // k
    group_means = []

    for i in range(k):
        start = i * group_size
        end = (i + 1) * group_size if i < k - 1 else n_samples
        group = samples[start:end]
        group_means.append(group.mean(dim=0))

    group_means = torch.stack(group_means, dim=0)
    median_pred = group_means.median(dim=0)[0]

    return median_pred
```

**修改predict()方法**:
```python
def predict(self, x_enc, x_mark_enc=None, ..., use_mom=True, mom_k=10):
    # ... 采样代码 ...

    # 计算均值
    if use_mom:
        # SimDiff的Median-of-Means方法（MSE降低8.3%）
        mean_pred = self.median_of_means(pred_samples, k=mom_k)
    else:
        # 简单均值（原始方法）
        mean_pred = pred_samples.mean(dim=0)

    return mean_pred, std_pred, pred_samples
```

**配置**:
- 默认启用：`use_mom=True`
- 分组数：`mom_k=10`（适合100个样本）
- 灵活切换：可通过 `use_mom=False` 恢复简单均值

**预期效果**:
- ✅ MSE降低5-10%（基于SimDiff论文）
- ✅ 对异常值更robust
- ✅ 保留temporal patterns

---

### ✅ Priority 3: 调整课程学习策略

**文件**: `exp/exp_diffusion_forecast.py`
**行数**: 230-268

**修改前**:
```python
def _get_loss_weights(self, epoch):
    if epoch < self.warmup_epochs:
        # Warmup: α从1.0线性降到0.5
        alpha = 1.0 - epoch / self.warmup_epochs * 0.5
    else:
        # 联合阶段: α=0.3（30% MSE + 70% Diffusion）
        alpha = 0.3  # ← 问题：MSE权重过低

    beta = 1.0 - alpha
    return alpha, beta
```

**修改后**:
```python
def _get_loss_weights(self, epoch):
    """
    修复策略：固定α=0.8（80% MSE + 20% Diffusion）

    理由：
    - Epoch 1 (α=1.0) 性能最佳
    - 后续α降低反而MSE上升
    - 扩散模型的MSE问题是已知trade-off
    - MSE权重过低会损害点预测性能
    """
    alpha = 0.8  # 固定80% MSE
    beta = 0.2   # 固定20% Diffusion

    return alpha, beta
```

**预期效果**:
- ✅ 点预测性能优先
- ✅ 训练更稳定
- ✅ MSE不会因课程学习而下降

---

## 验证测试

### ✅ 单元测试
**脚本**: `tests/test_fixes.py`
**状态**: ✅ 所有测试通过

```
测试 1: Median-of-Means 方法          ✓ 通过
测试 2: 损失权重调度                  ✓ 通过
测试 3: backbone_forward 方法调用     ✓ 通过
测试 4: MoM 集成到 predict() 方法     ✓ 通过
```

### ✅ 语法检查
```bash
python -m py_compile exp/exp_diffusion_forecast.py       ✓ 通过
python -m py_compile models/iTransformerDiffusionDirect.py ✓ 通过
```

---

## 实验脚本

**脚本**: `scripts/diffusion_forecast/ETT_script/iTransformerDiffusionDirect_ETTh1_fixed.sh`

### 运行方法
```bash
cd ~/projects/Time-Series-Library
bash scripts/diffusion_forecast/ETT_script/iTransformerDiffusionDirect_ETTh1_fixed.sh
```

### 配置参数
```bash
--task_name diffusion_forecast
--model iTransformerDiffusionDirect
--data ETTh1
--seq_len 96 --pred_len 96
--d_model 128 --e_layers 2
--train_epochs 30 --patience 5
--warmup_epochs 10
--diffusion_steps 1000
--beta_schedule cosine
--n_samples 100
--use_ddim --ddim_steps 50
--use_amp                    # 启用AMP（节省30-50%显存）
--parameterization v         # v-prediction（最稳定）
--des 'Fixed_MSE'            # 实验标识
```

---

## 预期性能

### 修复前（基线）
```
MSE:  0.7087
MAE:  0.5767
CRPS: 0.4961
校准度: 50%=0.499, 90%=0.858
```

### 修复后（目标）
```
MSE:  0.36-0.45  ← 降低40-50%
MAE:  0.45-0.55
CRPS: 0.40-0.45  ← 降低10%
校准度: 50%=0.48-0.52, 90%=0.88-0.92
```

### 对比确定性模型
```
PatchTST:    MSE 0.3771
iTransformer: MSE 0.3945
目标差距:     < 20% (vs 修复前 88-79%)
```

---

## 训练行为对比

### 修复前
```
Epoch 1: 训练损失0.765, 验证损失604.23 (错误！)
Epoch 2: 训练损失0.603, 验证损失605.48
Epoch 6: 训练损失0.506, 验证损失606.12
→ Early stopping触发（验证损失没有下降）
→ 最终MSE: 0.7087
```

### 修复后（预期）
```
Epoch 1: 训练损失0.765, 验证损失0.475 (正确！)
Epoch 5: 训练损失0.523, 验证损失0.396
Epoch 10: 训练损失0.461, 验证损失0.380
Epoch 15: 训练损失0.438, 验证损失0.365
Epoch 20: 训练损失0.425, 验证损失0.358
→ Early stopping在Epoch 20左右触发
→ 最终MSE: 0.36-0.45
```

---

## 修改文件清单

| 文件 | 修改内容 | 行数 | 状态 |
|------|---------|------|------|
| `exp/exp_diffusion_forecast.py` | 修复vali()损失计算 | 252-290 | ✅ |
| `exp/exp_diffusion_forecast.py` | 调整_get_loss_weights()权重 | 230-268 | ✅ |
| `models/iTransformerDiffusionDirect.py` | 添加median_of_means() | 625-656 | ✅ |
| `models/iTransformerDiffusionDirect.py` | predict()集成MoM | 677-685 | ✅ |
| `tests/test_fixes.py` | 验证测试脚本 | 新建 | ✅ |
| `scripts/.../iTransformerDiffusionDirect_ETTh1_fixed.sh` | 实验脚本 | 新建 | ✅ |

---

## 关键技术细节

### 1. 验证损失计算
```python
# backbone_forward返回
y_det: [B, pred_len, N]  # 确定性预测
z: [B, N, d_model]       # 条件特征
means, stdev             # 归一化统计量

# MSE计算
loss_mse = F.mse_loss(y_det, y_true)  # 真实的点预测MSE
```

### 2. Median-of-Means算法
```
输入: samples [100, B, pred_len, N]

步骤:
1. 分组: 100个样本 → 10组，每组10个
2. 组内均值: [10, B, pred_len, N]
3. 组间中位数: [B, pred_len, N]

优势:
- 对异常值robust（中位数）
- 保留均值的期望（分组）
- 不过度平滑temporal patterns
```

### 3. 损失权重固定策略
```python
# 每个batch的损失
loss_total = 0.8 * loss_mse + 0.2 * loss_diff

# 确保:
# 1. 点预测性能优先（80% MSE）
# 2. 学习不确定性（20% Diffusion）
# 3. 训练稳定（不随epoch变化）
```

---

## 理论支持

### SimDiff论文（2024年11月）
- 论文：[SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting](https://arxiv.org/html/2511.19256v1)
- 关键发现：Median-of-Means降低MSE 8.3%
- 适用场景：100个样本，k=10分组

### 扩散模型MSE问题（学术共识）
- 论文：[Diffusion Models for Time Series Forecasting: A Survey](https://arxiv.org/html/2507.14507)
- 引用：*"Probabilistic models optimized for distribution are seldom compared against strong point-forecasting baselines due to poor MSE scores."*
- 结论：点预测和概率预测存在inherent trade-off

### CSDI/TSDiff基准
- CSDI: CRPS降低40-65% vs baselines
- TSDiff: 自引导机制改善点预测
- 启示：扩散模型需要特殊的评估和训练策略

---

## 后续工作

### 如果MSE仍未达标（< 0.45）
1. **增大backbone容量**
   - d_model: 128 → 256
   - e_layers: 2 → 4

2. **延长训练**
   - train_epochs: 30 → 50
   - 观察是否继续下降

3. **尝试残差归一化调整**
   - 禁用残差归一化
   - 或使用全局统计量

### 如果CRPS需要优化
1. **调整扩散损失权重**
   - β: 0.2 → 0.3 (提高不确定性建模)

2. **增加采样步数**
   - ddim_steps: 50 → 100

3. **调整采样数量**
   - n_samples: 100 → 200

---

## 检查清单

运行实验后，检查以下指标：

### ✓ 训练行为
- [ ] 训练持续15-20 epoch（不会6个就停）
- [ ] 验证损失数值合理（0.3-0.6范围）
- [ ] 训练损失持续下降
- [ ] 无NaN/Inf

### ✓ 最终性能
- [ ] MSE < 0.45（vs 修复前0.71）
- [ ] MSE与iTransformer差距 < 20%（vs 修复前79%）
- [ ] CRPS < 0.45
- [ ] 校准度: 50%=0.48-0.52, 90%=0.88-0.92

### ✓ 日志检查
```bash
# 查看训练日志
cat checkpoints/*/log.txt

# 查看结果
cat result_diffusion_forecast.txt

# 查看checkpoint
ls -lh checkpoints/*/checkpoint.pth
```

---

## 技术联系人

如有问题，参考以下资源：

- **SimDiff代码**: [GitHub](https://github.com/simdiff/simdiff) (待发布)
- **CSDI代码**: https://github.com/ermongroup/CSDI
- **TSDiff代码**: https://github.com/amazon-science/unconditional-time-series-diffusion
- **调研文档**: `PLAN.md`

---

**实施完成日期**: 2026-01-22
**实施者**: Claude Code
**状态**: ✅ Ready for Experiment
