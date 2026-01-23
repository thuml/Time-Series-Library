# Phase 1 基础优化实施计划

> **目标**: 解决 iTransformerDiffusionDirect 模型的核心问题，提升性能和稳定性
> **预期效果**: MSE 从 0.60 降至 0.42-0.45, 训练稳定性显著提升

---

## 任务概览

| 任务 | 优先级 | 预期收益 | 代码改动量 |
|------|:------:|----------|-----------|
| 1. v-prediction 切换 | ⭐⭐⭐ | MSE -15%, 无需 clamp | ~20 行修改 |
| 2. 端到端联合训练 | ⭐⭐⭐ | MSE -10%, 梯度连通 | ~150 行重构 |
| 3. 时序感知损失 | ⭐⭐ | CRPS -10% | ~100 行新增 |
| 4. 修复已知 Bug | ⭐⭐⭐ | 运行稳定 | ~10 行修改 |

---

## 任务 1: v-prediction 参数化切换

### 1.1 问题分析
当前模型默认使用 `x0` 参数化，在高噪声时间步（t → T）预测不稳定，需要 `clamp(-3, 3)` 强制稳定。

### 1.2 解决方案
模型代码 `models/iTransformerDiffusionDirect.py` 已支持 v-prediction，只需：

**修改文件**: `run.py` 或训练脚本
```bash
# 添加命令行参数
--parameterization v
```

**修改文件**: `models/iTransformerDiffusionDirect.py:76`
```python
# 将默认值从 'x0' 改为 'v'
self.parameterization = getattr(configs, "parameterization", "v")  # 默认使用 v
```

### 1.3 验证方式
- 移除 `sample_ddpm` 中的 `clamp` 后模型仍稳定
- 预测 std 从 0.73 提升至接近 1.0

---

## 任务 2: 端到端联合训练重构

### 2.1 问题分析
当前两阶段训练存在致命问题：
- Stage 2 调用 `model.freeze_encoder()` 后，扩散损失的梯度无法流回 backbone
- Backbone 只学习确定性预测的特征，无法学习对扩散有利的特征表示

### 2.2 解决方案
**重构 `exp/exp_diffusion_forecast.py`**:

1. **移除两阶段分离**，改为端到端训练
2. **实现课程学习权重调度**：
   - 前 10 epochs: MSE 权重高（warmup）
   - 后续 epochs: 逐渐增加扩散损失权重
3. **保持 backbone 可训练**，不再冻结

### 2.3 具体修改

**新增方法**: `train_end_to_end()`
```python
def train_end_to_end(self, setting):
    """端到端联合训练（替代两阶段训练）"""
    # 课程学习权重调度
    def get_loss_weights(epoch, warmup=10, total=50):
        if epoch < warmup:
            alpha = 1.0 - epoch / warmup * 0.5  # 1.0 → 0.5
        else:
            alpha = 0.3  # 30% MSE + 70% Diffusion
        return alpha, 1 - alpha

    # 训练循环（backbone + diffusion 联合优化）
    for epoch in range(total_epochs):
        alpha, beta = get_loss_weights(epoch)
        for batch in train_loader:
            loss, loss_dict = model.forward_loss(..., stage='joint')
            # loss = alpha * loss_mse + beta * loss_diff
            loss.backward()  # 梯度可以流回 backbone！
```

**修改 `train()` 方法**：
```python
def train(self, setting):
    # 根据配置选择训练模式
    if self.training_mode == 'end_to_end':
        return self.train_end_to_end(setting)
    else:
        # 保留原有两阶段训练作为备选
        self.train_stage1(...)
        self.train_stage2(...)
```

**修改优化器配置**：
```python
def _select_optimizer_end_to_end(self):
    """端到端训练优化器（分组学习率）"""
    param_groups = [
        {'params': self.model.enc_embedding.parameters(), 'lr': 1e-4},
        {'params': self.model.encoder.parameters(), 'lr': 1e-4},
        {'params': self.model.projection.parameters(), 'lr': 1e-4},
        {'params': self.model.denoise_net.parameters(), 'lr': 1e-4},
    ]
    return optim.AdamW(param_groups, weight_decay=0.01)
```

---

## 任务 3: 时序感知损失函数

### 3.1 问题分析
当前只使用简单 MSE 损失，未考虑时序数据特性：
- 未捕捉趋势（一阶差分）
- 未保持周期性（频域）
- 未维护变量间相关性

### 3.2 解决方案
**新增文件**: `utils/ts_losses.py`

```python
class TimeSeriesAwareLoss(nn.Module):
    """时序感知复合损失函数"""

    def __init__(self, lambda_point=1.0, lambda_trend=0.1,
                 lambda_freq=0.1, lambda_corr=0.05):
        super().__init__()
        self.lambdas = {
            'point': lambda_point,
            'trend': lambda_trend,
            'freq': lambda_freq,
            'corr': lambda_corr
        }

    def point_loss(self, pred, target):
        """点级 MSE"""
        return F.mse_loss(pred, target)

    def trend_loss(self, pred, target):
        """趋势损失：一阶差分 MSE"""
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return F.mse_loss(pred_diff, target_diff)

    def frequency_loss(self, pred, target):
        """频域损失：FFT 幅度谱 MSE"""
        pred_fft = torch.fft.rfft(pred, dim=1).abs()
        target_fft = torch.fft.rfft(target, dim=1).abs()
        return F.mse_loss(pred_fft, target_fft)

    def correlation_loss(self, pred, target):
        """相关性损失：变量间相关矩阵 MSE"""
        def compute_corr(x):
            B, T, N = x.shape
            x_centered = x - x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True) + 1e-5
            x_norm = x_centered / x_std
            return torch.bmm(x_norm.transpose(1, 2), x_norm) / T

        return F.mse_loss(compute_corr(pred), compute_corr(target))

    def forward(self, pred, target):
        losses = {
            'point': self.point_loss(pred, target),
            'trend': self.trend_loss(pred, target),
            'freq': self.frequency_loss(pred, target),
            'corr': self.correlation_loss(pred, target)
        }

        total = sum(self.lambdas[k] * v for k, v in losses.items())
        return total, {f'loss_{k}': v.item() for k, v in losses.items()}
```

### 3.3 集成方式
**修改 `models/iTransformerDiffusionDirect.py` 的 `forward_loss` 方法**：
```python
from utils.ts_losses import TimeSeriesAwareLoss

# 在 __init__ 中初始化
self.ts_loss = TimeSeriesAwareLoss() if getattr(configs, 'use_ts_loss', False) else None

# 在 forward_loss 中使用
if self.ts_loss is not None:
    loss_mse, ts_loss_dict = self.ts_loss(y_det, y_true)
else:
    loss_mse = F.mse_loss(y_det, y_true)
```

---

## 任务 4: 修复已知 Bug

### 4.1 Bug 1: `_select_optimizer_stage2` 引用不存在的属性

**文件**: `exp/exp_diffusion_forecast.py:184`
```python
# 错误: 引用 residual_normalizer（不存在）
{'params': self.model.residual_normalizer.parameters(), 'lr': self.stage2_lr * 10},

# 修复: 应该使用 output_normalizer
{'params': self.model.output_normalizer.parameters(), 'lr': self.stage2_lr * 10},
```

### 4.2 Bug 2: 采样函数中多余的 clamp（v-prediction 不需要）

**文件**: `models/iTransformerDiffusionDirect.py`

当使用 v-prediction 时，`sample_ddpm` 和 `sample_ddim` 中的 `clamp` 可以条件性移除：
```python
x0_pred = self.predict_x0_from_output(model_output, x, t_batch)
if self.parameterization == 'x0':
    x0_pred = torch.clamp(x0_pred, -3.0, 3.0)  # 只对 x0 参数化需要
```

---

## 实施顺序

```
Step 1: 修复 Bug (任务 4)
   └── 修改 exp/exp_diffusion_forecast.py 中的属性名错误
   └── 测试两阶段训练是否正常运行

Step 2: v-prediction 切换 (任务 1)
   └── 修改默认参数化类型
   └── 条件性移除 clamp
   └── 验证稳定性

Step 3: 时序感知损失 (任务 3)
   └── 新建 utils/ts_losses.py
   └── 集成到模型中
   └── 添加命令行参数控制

Step 4: 端到端训练重构 (任务 2)
   └── 实现 train_end_to_end 方法
   └── 实现课程学习权重调度
   └── 实现分组学习率优化器
   └── 添加训练模式切换参数
```

---

## 测试验证

### 验证脚本
```bash
# 1. 快速测试（小 epoch）
python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model iTransformerDiffusionDirect \
  --data ETTh1 \
  --seq_len 96 --pred_len 96 \
  --parameterization v \
  --training_mode end_to_end \
  --use_ts_loss \
  --train_epochs 5 \
  --batch_size 32

# 2. 完整训练
python run.py \
  --task_name diffusion_forecast \
  --is_training 1 \
  --model iTransformerDiffusionDirect \
  --data ETTh1 \
  --seq_len 96 --pred_len 96 \
  --parameterization v \
  --training_mode end_to_end \
  --use_ts_loss \
  --train_epochs 50 \
  --warmup_epochs 10 \
  --batch_size 32 \
  --use_amp
```

---

## 新增/修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `utils/ts_losses.py` | 新建 | 时序感知损失函数 |
| `exp/exp_diffusion_forecast.py` | 重构 | 端到端训练 + Bug 修复 |
| `models/iTransformerDiffusionDirect.py` | 修改 | v-prediction 默认 + 条件 clamp |
| `run.py` | 修改 | 添加新参数 |

---

## 预期结果

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| MSE | 0.5995 | 0.42-0.45 | -25~30% |
| CRPS | 0.495 | 0.35-0.40 | -20~30% |
| 预测 std | 0.73 | ~1.0 | 更准确的不确定性 |
| 训练稳定性 | 需要 clamp | 无需 clamp | ✅ |
| 梯度流动 | 断开 | 连通 | ✅ |
