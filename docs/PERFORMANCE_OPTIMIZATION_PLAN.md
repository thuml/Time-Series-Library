# iTransformerDiffusion 性能优化计划

**创建日期**: 2025-12-29
**状态**: 待实施

---

## 1. 当前性能瓶颈分析

### 1.1 测试阶段耗时分析 (ETTh1 快速测试)

| 阶段 | 时间 | 占比 |
|------|------|------|
| Stage 1 训练 | 8s | 0.9% |
| Stage 2 训练 | 32s | 3.6% |
| **测试采样** | **14min** | **95.5%** |
| 总计 | 14m37s | 100% |

**结论**: 采样是最大瓶颈，占总时间 95%+

### 1.2 采样瓶颈根因

当前 `sample_ddpm` 和 `sample_ddim` 实现：

```python
# 当前实现 (顺序采样)
all_samples = []
for _ in range(n_samples):  # 外层循环: 100次
    x = torch.randn(B, N, pred_len)
    for t in range(timesteps):  # 内层循环: 50-1000次
        noise_pred = denoise_net(x, t, z)  # GPU利用率低
        x = update(x, noise_pred)
    all_samples.append(x)
```

**问题**:
- 100 个样本顺序生成，无法并行
- GPU 每次只处理 batch_size=16 的数据
- 实际 GPU 利用率 < 20%

---

## 2. 优化方案

### 优化 1: 批量并行采样 ⭐⭐⭐ (最高优先级)

**原理**: 将 n_samples 合并到 batch 维度，一次 U-Net 前向处理所有样本

**修改文件**: `models/iTransformerDiffusion.py`

**实现方案**:

```python
@torch.no_grad()
def sample_ddpm_batch(self, z, n_samples=1):
    """批量并行 DDPM 采样"""
    B, _, d = z.shape
    N = self.n_vars
    device = z.device

    # 扩展 z: [B, N, d] → [n_samples*B, N, d]
    z_expanded = z.unsqueeze(0).expand(n_samples, -1, -1, -1)
    z_expanded = z_expanded.reshape(n_samples * B, N, d)

    # 批量初始化: [n_samples*B, N, pred_len]
    x = torch.randn(n_samples * B, N, self.pred_len, device=device)

    # 单次循环处理所有样本
    for t in reversed(range(self.timesteps)):
        t_batch = torch.full((n_samples * B,), t, device=device, dtype=torch.long)
        noise_pred = self.denoise_net(x, t_batch, z_expanded)
        # DDPM 更新...
        x = update(x, noise_pred, t)

    # 重组: [n_samples*B, N, T] → [n_samples, B, N, T]
    return x.reshape(n_samples, B, N, self.pred_len)
```

**预期效果**:
- 采样速度提升: **5-10x**
- GPU 利用率: 20% → 80%+
- 显存增加: ~n_samples 倍 (需要权衡)

**显存控制策略**:
```python
def sample_ddpm_chunked(self, z, n_samples=100, chunk_size=10):
    """分块采样，平衡速度和显存"""
    all_samples = []
    for i in range(0, n_samples, chunk_size):
        chunk_n = min(chunk_size, n_samples - i)
        samples = self.sample_ddpm_batch(z, chunk_n)
        all_samples.append(samples)
    return torch.cat(all_samples, dim=0)
```

---

### 优化 2: AMP (FP16) 训练支持 ⭐⭐⭐

**原理**: 使用混合精度训练，减少显存占用，加速计算

**修改文件**: `exp/exp_diffusion_forecast.py`

**实现方案**:

```python
# 初始化
if self.args.use_amp:
    self.scaler = torch.cuda.amp.GradScaler()

# Stage 1 训练循环
def train_stage1(self, setting):
    for epoch in range(self.args.stage1_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    loss, loss_dict = self.model.forward_loss(
                        batch_x, batch_x_mark, y_true, stage='warmup'
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss, loss_dict = self.model.forward_loss(...)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(...)
                optimizer.step()
```

**预期效果**:
- 训练速度提升: **1.3-1.5x**
- 显存减少: **30-50%**
- 精度影响: 几乎无影响

---

### 优化 3: DDIM 参数优化 ⭐⭐

**原理**: 减少默认采样步数，预计算时间步序列

**修改文件**: `models/iTransformerDiffusion.py`, `run.py`

**实现方案**:

```python
# 1. 调整默认参数
# run.py
parser.add_argument('--ddim_steps', type=int, default=20)  # 50 → 20
parser.add_argument('--n_samples', type=int, default=50)   # 100 → 50

# 2. 预计算 DDIM 时间步 (避免重复计算)
class Model(nn.Module):
    def __init__(self, configs):
        # ...
        self._ddim_timesteps_cache = {}

    def _get_ddim_timesteps(self, ddim_steps):
        if ddim_steps not in self._ddim_timesteps_cache:
            step_size = self.timesteps // ddim_steps
            self._ddim_timesteps_cache[ddim_steps] = list(
                range(0, self.timesteps, step_size)
            )[::-1]
        return self._ddim_timesteps_cache[ddim_steps]
```

**预期效果**:
- 默认采样时间减少: **2-2.5x**
- CRPS 质量: 影响 < 5%

---

### 优化 4: CRPS 向量化计算 ⭐⭐

**原理**: 用向量化操作替代 Python 循环

**修改文件**: `exp/exp_diffusion_forecast.py`

**当前实现**:
```python
def crps_score(samples, y_true):
    crps = 0.0
    for i in range(n_samples):  # Python 循环
        indicator = (samples_sorted[i] <= y_true).float()
        crps += (indicator - (i+1)/n_samples) ** 2
```

**优化后**:
```python
def crps_score_vectorized(samples, y_true):
    n_samples = samples.shape[0]
    samples_sorted = torch.sort(samples, dim=0)[0]

    # 向量化: 一次计算所有样本
    indices = torch.arange(1, n_samples + 1, device=samples.device)
    indices = indices.reshape(-1, 1, 1, 1)  # [n, 1, 1, 1]
    ecdf = indices.float() / n_samples

    indicator = (samples_sorted <= y_true.unsqueeze(0)).float()
    crps = ((indicator - ecdf) ** 2).sum(dim=0).mean() / n_samples

    return crps.item()
```

**预期效果**:
- CRPS 计算加速: **3-5x**
- 对总时间影响: < 1% (非瓶颈)

---

### 优化 5: 采样进度条 ⭐

**原理**: 添加 tqdm 进度条，提升用户体验

**修改文件**: `models/iTransformerDiffusion.py`

```python
from tqdm import tqdm

@torch.no_grad()
def sample_ddim(self, z, n_samples=1, ddim_steps=50, eta=0.0, verbose=True):
    # ...
    timesteps = self._get_ddim_timesteps(ddim_steps)

    iterator = tqdm(timesteps, desc='DDIM Sampling', disable=not verbose)
    for i, t in enumerate(iterator):
        # 采样步骤...
```

---

## 3. 实施计划

### Phase 1: 核心优化 (预计 2 小时)

| 任务 | 优先级 | 预计时间 |
|------|--------|---------|
| 批量并行采样 | P0 | 45 min |
| AMP 训练支持 | P0 | 30 min |
| 单元测试更新 | P0 | 30 min |

### Phase 2: 辅助优化 (预计 1 小时)

| 任务 | 优先级 | 预计时间 |
|------|--------|---------|
| DDIM 参数优化 | P1 | 15 min |
| CRPS 向量化 | P1 | 20 min |
| 进度条添加 | P2 | 10 min |

### Phase 3: 验证 (预计 30 分钟)

| 任务 | 预计时间 |
|------|---------|
| 性能基准测试 | 15 min |
| 对比报告生成 | 15 min |

---

## 4. 预期总体效果

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 测试时间 (ETTh1) | 14 min | 2-3 min | **5-7x** |
| 训练时间 | 40s | 30s | 1.3x |
| GPU 利用率 | ~20% | ~80% | 4x |
| 显存占用 | 4GB | 3GB (AMP) | -25% |

---

## 5. 修改文件清单

| 文件 | 修改类型 | 内容 |
|------|---------|------|
| `models/iTransformerDiffusion.py` | 重构 | sample_ddpm/ddim 批量化 |
| `exp/exp_diffusion_forecast.py` | 增强 | AMP 支持, CRPS 向量化 |
| `run.py` | 参数调整 | 默认值优化 |
| `tests/test_iTransformerDiffusion.py` | 更新 | 新采样方法测试 |

---

## 6. 风险和缓解

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| 批量采样显存溢出 | 中 | 实现分块采样 (chunk_size) |
| AMP 精度问题 | 低 | 保留 FP32 回退选项 |
| DDIM 步数过少质量下降 | 低 | 保留可配置参数 |

---

## 7. 验收标准

- [ ] 批量采样通过单元测试
- [ ] AMP 训练收敛正常
- [ ] 测试时间 < 5 分钟 (ETTh1)
- [ ] CRPS 变化 < 5%
- [ ] 无新增 bug

---

**准备就绪，可以开始实施。**
