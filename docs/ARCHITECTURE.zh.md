# 逐层 CEEMDAN 分解的实现约定

## 为什么本版符合骨干融合

前一版 V2 从原始历史提取 CEEMDAN 分量，作为适配器信息进入 PatchTST。本版删除该依赖，直接在编码层残差流里做分解。每次前向都对当前隐状态计算，参数更新之后的隐状态会重新分解，没有预计算隐状态或沿用过时的信号缓存。

输入 `[B,L,C]` 先按官方 PatchTST 做 RevIN、末尾复制填充、重叠分块、共享线性嵌入和位置编码，变成 `[B*C,N,D]`。每个原始变量使用相同网络参数，注意力只沿 N 轴计算。

第 l 层令输入为 h，聚合更新为 A(h)，执行：

1. `u = h + Dropout(A(h))`。
2. `(s1, r1) = CEEMDAN(u)`；`a = Norm(s1)` 使用官方 PatchTST 的归一化。
3. `v = a + Dropout(FFN(a))`。
4. `(s2, r2) = CEEMDAN(v)`；`h_next = Norm(s2)`。
5. `trend_forecast_l = FlattenHead_l(r1+r2)`。

最终 `y_normalized = SeasonalFlattenHead(h_last) + sum_l(trend_forecast_l)`，之后只做一次 RevIN 逆变换，避免重复加均值或仿射偏置。

这借鉴 Autoformer 的逐层分解，以及 decoder 中趋势预测逐层相加的机制，但本身仍是无 decoder 的 PatchTST 直接预测骨干，不是完整 Autoformer encoder-decoder 的逐行替换。

## CEEMDAN 的含义

使用 `EMD-signal==1.10.0` 的 `PyEMD.CEEMDAN`，固定 seed、`parallel=False`。每条隐特征分别分解，振荡项为所有提取 IMF 之和，余项为输入减振荡项。高频 IMF 不丢弃，也不强制按 IMF 编号跨样本对齐。

默认 `max_imf=3` 是计算预算上的截断。最终余项仍可能含有振荡，因此文档采用“余项/趋势路径”，不声称它是纯物理趋势。隐特征上的分解也不等同于原始观测变量的物理季节性解释。

常量和单调信号直接作为余项，不强行给无振荡信号注入 IMF。序列至少需要 3 个 patch。对近乎常量信号使用标准差 `<1e-10` 的保护。非有限输入或分解结果直接报错，不静默改成移动平均。

相同 seed 对每条序列重置，噪声不依赖 batch、通道编号或隐藏特征编号；因此重排或拆分 batch 不改变对应分解。只复用完全相同固定噪声的 EMD，测试将该优化与未修改 PyEMD 的真实分解逐条比较。不会缓存当前隐状态的分解结果。

## 不可微问题：明确采用固定余项

PyEMD 是 NumPy/CPU 算法，不能直接通过 PyTorch 自动微分。实现的前向是真实 CEEMDAN，反向采用：

```text
r = stop_gradient(CEEMDAN_remainder(x))
s = x - r
ds/dx = I
dr/dx = 0
```

这是一种人为指定的代理梯度，**不是 CEEMDAN 的真实导数**。振荡残差路径让注意力、嵌入、位置编码、FFN 继续接收预测 MSE 的梯度；余项预测头能学习其投影权重，但余项路径不能通过 CEEMDAN 向此前层传梯度。不存在可训练的“CEEMDAN 参数”。本版没有增加教师训练、可学习分解替代器或与原始输入输出的混合旁路。

移动平均消融使用其自然导数，因此它与 CEEMDAN 的差异同时包含前向分解和梯度机制，不能仅凭该消融把收益全部归因于更精细的分解。若后续研究要求对分解本身求导，需要另行定义可微近似或可学习分解器。

## 时序依赖与通道独立

默认 `hybrid` 用每层一个可学习标量：`(1-sigmoid(g))*SelfAttention + sigmoid(g)*AutoCorrelation`，初始 g=0。自注意力复用官方 PatchTST 的多头注意力及跨层 residual attention scores；周期路径按 Autoformer 的 FFT 相关发现与 top-k 环形时延聚合设计实现，时延单位是 patch stride。

周期 top-k 在训练和推理均逐实例选择。官方 Autoformer 训练时的 batch 共享时延会在把通道折入 batch 后间接引入通道依赖，因此这里有意改为逐实例。FFT 显式指定 N，支持奇数 patch 数。环形聚合只在历史 patch 内取值，不访问预测时点之后的数据。

默认保留 PatchTST 的 BatchNorm：训练统计量会在折叠后的 batch/通道之间共享，这是官方 PatchTST 本身的设计。若需要连训练统计量也严格逐通道独立，使用 `--encoder_norm LayerNorm`。推理状态下两种归一化都不依赖其他输入通道的当前值。所有预测头共享通道参数，融合模型拒绝 `individual=1`。

## 训练机制与评估

使用官方监督训练框架，模型只收到 batch_x。Adam 优化最终预测 MSE，支持官方学习率策略，推荐 `lradj=TST` 使用逐步 OneCycleLR；只按验证集选择 checkpoint。避免每轮测试集前向，最优模型选择完才测试。验证和测试不打乱、不丢最后不足一个 batch 的样本，指标按完整拼接的预测重新计算。

`max_train_steps`、`max_eval_steps` 仅用于调试，默认 0 表示不限制。它们不应出现在完整效果比较里。完整比较应在看测试结果前固定配置，两个原模型分别按各自合理超参数训练，使用多个种子，并报告误差与时间开销。

## 计算成本

若层数 E、batch B、通道 C、隐藏宽度 D，则每次前向至多需 **2×E×B×C×D 次 CEEMDAN**，每次长度 N，且每次内部还包含多次噪声试验和 EMD 筛分。默认完整脚本 E=2、B=4、C=7、D=16，至多 1,792 次/前向；smoke E=2、B=2、C=1、D=8，至多 64 次/前向。单调/常量序列会跳过。

神经网络可以在 CUDA 上运行，但每个分解点都需 CPU 数据往返同步，GPU 不能消除 CEEMDAN 瓶颈。本版明确保留真实逐特征、逐层分解，不悄悄用低秩替代、移动平均、每隔若干步更新或离线特征缓存降低成本。可以先用较窄骨干、较小 batch、较少 trials 验证结构，再逐步增加预算；较少 trials 也可能改变分解质量。

当前 Mac CPU 上完整脚本配置的一次多变量前向实测约 10.35 秒、反向约 0.019 秒，实际 1,792 次 CEEMDAN 全部执行。记录见 `multivariate_check.json`，这是单批功能检查，不代表完成训练或测得了稳定吞吐率。

## 来源

- [PatchTST 官方实现](https://github.com/yuqinie98/PatchTST)，本地版本 commit 见 UPSTREAM.json。
- [Autoformer 官方实现](https://github.com/thuml/Autoformer)，参考逐层分解、周期聚合和 decoder 余项累积。
- [PyEMD CEEMDAN 文档](https://pyemd.readthedocs.io/en/latest/ceemdan.html)，固定种子与不可微 NumPy 后端。
