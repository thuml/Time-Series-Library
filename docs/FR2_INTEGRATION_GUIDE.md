# FR2 (Frequency-aware Residual) 集成指南

## 概述

FR2（Frequency-aware Residual）是一个频域感知的残差连接模块，用于增强扩散模型的频域表达能力。

## 何时使用 FR2

根据 Phase 2 基线训练结果，**仅在以下情况下考虑集成 FR2**：

1. **频域损失明显偏高**：横向对比发现 `loss_freq` 显著高于其他模型
2. **周期性预测不准确**：可视化显示模型无法捕捉数据的周期性模式
3. **总体性能有提升空间**：CRPS 指标距离目标较远

## 模块位置

FR2 模块已添加到：`layers/Diffusion_layers.py`

验证脚本位于：`tests/validate_fr2.py`

## 集成步骤

### 方案 1: 在 UNet1D bottleneck 集成（推荐）

这是最简单的集成方式，在 bottleneck 之后添加 FR2：

**修改文件**: `layers/Diffusion_layers.py`

**原始代码** (约 line 385-387):
```python
# Bottleneck
h = self.bottleneck_res(h, cond)
h = self.bottleneck_attn(h, z)
```

**修改后代码**:
```python
# Bottleneck
h = self.bottleneck_res(h, cond)
h = self.bottleneck_attn(h, z)

# FR2: 频域感知残差调制（可选）
if hasattr(self, 'fr2'):
    h = self.fr2(h, z)
```

**在 UNet1D.__init__ 中添加** (约 line 302-360):
```python
# 在 __init__ 末尾，final_conv 定义之前添加:

# FR2 (可选，根据配置决定是否启用)
if use_fr2:  # 新增配置参数
    self.fr2 = FrequencyAwareResidual(
        channels=channels[-1],  # bottleneck 的通道数
        d_model=d_model,
        n_freqs=10  # 可配置
    )
```

### 方案 2: 在每个 UpBlock 集成（更激进）

如果方案 1 效果不明显，可以在每个上采样块中添加 FR2：

**修改**: `UpBlock1D` 类的 forward 方法

```python
# 在每个 UpBlock1D 的 forward 末尾添加:
if self.use_fr2:
    h = self.fr2(h, z)
```

## 配置参数

建议在 `run.py` 中添加以下参数：

```python
# 在 ArgumentParser 中添加:
parser.add_argument('--use_fr2', action='store_true',
                    help='启用 FR2 频域感知残差模块')
parser.add_argument('--fr2_n_freqs', type=int, default=10,
                    help='FR2 频率分辨率')
```

## 训练脚本示例

```bash
python run.py \
  --task_name diffusion_forecast \
  --model iTransformerDiffusionDirect \
  --data ETTh1 \
  --seq_len 96 --pred_len 96 \
  --d_model 128 --e_layers 2 \
  --use_fr2 \              # 启用 FR2
  --fr2_n_freqs 10 \       # 频率分辨率
  ...
```

## 验证测试

运行以下命令验证 FR2 模块正常工作：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tslib
python tests/validate_fr2.py
```

**预期输出**:
```
============================================================
FR2 (Frequency-aware Residual) 模块验证
============================================================
...
所有测试通过！FR2 模块工作正常 ✓
============================================================
```

## 消融实验

集成 FR2 后，建议进行消融实验：

1. **Baseline**: 不使用 FR2
2. **+ FR2**: 启用 FR2
3. **对比指标**:
   - MSE, MAE, RMSE (点预测)
   - CRPS (概率预测)
   - Frequency Loss (频域损失)
   - 训练时间

## 性能影响

**参数量**: 约增加 `d_model * n_freqs * 2 + d_model * channels` 个参数

**计算量**: 每次前向传播增加 1 次 FFT 和 1 次 IFFT

**显存**: 几乎无影响（FFT 不需要额外显存）

**预期提升**:
- 频域损失: -10% ~ -20%
- CRPS: -5% ~ -10%
- MSE: -3% ~ -8%

## 注意事项

1. **先完成基线训练**: 确保有稳定的基线性能作为对比
2. **数据驱动决策**: 根据横向对比结果决定是否需要 FR2
3. **单独验证**: FR2 应该单独验证其效果，不要与其他优化同时实施
4. **参数调优**: `n_freqs` 参数需要根据数据的频域特性调整
   - 低频主导数据: `n_freqs=5-10`
   - 多周期数据: `n_freqs=15-20`

## 故障排查

### 问题 1: FFT 错误
```
RuntimeError: MKL FFT error: ...
```
**解决**: 已在代码中添加 `.contiguous()` 修复

### 问题 2: NaN/Inf
```
Output contains NaN
```
**解决**: 检查 `amp_mod` 和 `phase_mod` 的范围，可能需要添加剪切

### 问题 3: 性能下降
**原因**: FR2 可能不适合当前数据
**解决**: 调整 `n_freqs` 或禁用 FR2

## 决策流程图

```
Phase 2 基线训练完成
    │
    ├─ 频域损失高？
    │   ├─ 是 → 集成 FR2
    │   └─ 否 ↓
    │
    ├─ 周期性预测差？
    │   ├─ 是 → 集成 FR2
    │   └─ 否 ↓
    │
    └─ 整体性能接近目标？
        ├─ 是 → 无需 FR2
        └─ 否 → 尝试其他优化（温度缩放、CCFB）
```

## 参考文献

FR2 设计灵感来源：
- [FiLM: Visual Reasoning with Feature-wise Linear Modulation](https://arxiv.org/abs/1709.07871)
- [FNO: Fourier Neural Operator](https://arxiv.org/abs/2010.08895)

---

**最后更新**: 2026-01-22
**状态**: 代码已实现，测试通过，等待基线结果决定是否集成
