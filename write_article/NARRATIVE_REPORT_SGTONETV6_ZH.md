# 叙事报告：面向短时域提升机故障状态预测的边界约束稀有故障触发

> **这是用于 Workflow 3 的 SGTONetV6 专用 NARRATIVE_REPORT.md。** 结构参照 `NARRATIVE_REPORT_EXAMPLE.md`，但内容替换为当前 Hoister 私有数据集、SGTONetV6 实现过程和已完成实验结果。

## 核心故事

本文研究工业提升机超速过程中的固定短时域未来状态预测。该任务不是普通的事后故障分类：模型需要根据当前多变量传感器窗口预测下一短时域运行状态。真正困难不只是平均分类性能，而是状态转移边界附近的严重稀有类别不均衡。在 Hoister 私有数据集中，二级退化状态，即标签 `9`，在 37,417 个时间戳中只出现 185 次，但它具有安全意义，因为它位于故障过程之前或故障过程附近。

强时序分类器可以取得较高总体准确率，却完全漏检这个稀有状态。在当前 `label_shift=1` 实验中，iTransformer 的准确率达到 `0.8175`，但 class-9 F1 为 `0.0000`。DLinear、TimesNet、PatchTST 和保守版 SGTONetV4 的 class-9 F1 也都是 `0.0000`。

我们提出 **SGTONetV6**，即一种 shift-aware graph and trigger oriented network。它把两类行为拆开：一条保守的多类别未来状态分类路径用于常见状态预测，另一条受边界约束的稀有故障触发路径用于 class `9` 恢复。当前实现使用 SGTO 系列的 patch temporal encoder、图约束未来状态修正、prototype 辅助未来分类器、patch-attentive rare context 模块，以及由边界和前驱状态语义约束的推理期 rare override 规则。SGTONetV6 在短时域设置下提升了 macro-F1 和 fault macro-F1，更重要的是恢复了稀有 class-9 状态。

## Claims

1. **短时域 Hoister 未来状态预测存在 rare-boundary collapse 失效模式**：标准分类器可以在整体指标上表现较好，但完全不预测稀有二级退化状态。在 Hoister 私有任务的 `label_shift=1` 设置中，所有非 trigger baseline 的 class-9 F1 都是 `0.0000`。

2. **SGTONetV6 通过拆分基础分类器和稀有触发器来恢复 class `9`**：完整的 SGTONetV6DualOverride 达到 macro-F1 `0.6233`、balanced accuracy `0.6731`、fault macro-F1 `0.5411`、class-9 F1 `0.5556`；而最强 baseline iTransformer 的 macro-F1 为 `0.6185`，class-9 F1 为 `0.0000`。

3. **性能提升来自边界约束触发机制，而不是单纯更强的 backbone**：移除 rare override 后 class-9 F1 降为 `0.0000`；移除 boundary constraint 后 class-9 F1 降为 `0.0158`；把 patch-attentive rare context 替换成 mean context 后 class-9 F1 降为 `0.2317`。

4. **当前证据只支持短时域 claim，不支持宽泛的 multi-horizon claim**：在 `label_shift=3` 下，PatchTST 的 macro-F1 为 `0.5877`、class-9 F1 为 `0.1919`，而 SGTONetV6DualOverride 的 macro-F1 为 `0.5006`、class-9 F1 为 `0.1070`。这应作为局限性明确披露。

## Experiments

### Setup
- **模型**：SGTONetV6DualOverride，实现位于 `models/SGTONetV6.py`
- **数据**：Hoister 私有超速数据集，27 个 CSV 文件，37,417 个时间戳，20 列
- **目标列**：`running_state_five_class`
- **丢弃列**：`id`、`time`、`JianSuDuan_ChaoSu`、`running_state_class`、`running_state_five_class`
- **状态**：标签 `1` 表示停机，标签 `5` 表示正常，标签 `7` 表示一级退化，标签 `9` 表示二级退化，标签 `3` 表示故障发生
- **窗口设置**：`seq_len=96`，`window_step=8`，`window_label_mode=last`
- **预测时域**：主证据使用 `label_shift=1`
- **划分方式**：文件级 split seeds `14`、`22`、`30`
- **Baselines**：DLinear、TimesNet、iTransformer、PatchTST、SGTONetV4Conservative
- **指标**：accuracy、macro-F1、weighted F1、balanced accuracy、fault macro-F1、class-9 precision/recall/F1

### 数据集概况

数据集位于 `dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579`。它包含 27 个 CSV 文件和 37,417 行数据。每个 CSV 有以下 20 列：

```text
id, time, SuDuMoNiLiang, FPLCSuDu, BianmaQiSuDu, CSJSuDu,
LGuanLongShenDu, FPLCShenDu, BianmaQiShenDu, DianshuDianliu1,
DianshuDianliu2, LiCi_Current, ZhiDongPressure, ZhuLingDWQ1,
ZhaBaDWQ1, WZhuJiLiang, WFuJiLiang, JianSuDuan_ChaoSu,
running_state_class, running_state_five_class
```

五分类标签分布如下：

| Label | 含义 | Count |
|---:|---|---:|
| 1 | 停机 | 10,959 |
| 5 | 正常运行 | 15,695 |
| 7 | 一级退化 | 5,364 |
| 9 | 二级退化 | 185 |
| 3 | 故障发生 | 5,214 |

**解释**：class `9` 占所有时间戳不到 0.5%。因此，class-9 恢复是本文最核心的安全导向评价问题。

### Experiment 1: 主短时域对比（Table 1, Figures 1-2）

在 `label_shift=1` 未来状态预测任务上，对比 SGTONetV6DualOverride 与 DLinear、TimesNet、iTransformer、PatchTST、SGTONetV4Conservative。

| Model | Accuracy | Macro-F1 | Balanced Acc. | Fault Macro-F1 | Class9 F1 |
|---|---:|---:|---:|---:|---:|
| **SGTONetV6DualOverride** | 0.7102 | **0.6233** | **0.6731** | **0.5411** | **0.5556** |
| iTransformer | **0.8175** | 0.6185 | 0.6594 | 0.4615 | 0.0000 |
| DLinear | 0.7895 | 0.5961 | 0.6466 | 0.4426 | 0.0000 |
| TimesNet | 0.7958 | 0.5893 | 0.6207 | 0.4275 | 0.0000 |
| PatchTST | 0.7559 | 0.5687 | 0.6154 | 0.4194 | 0.0000 |
| SGTONetV4Conservative | 0.7630 | 0.5605 | 0.5961 | 0.3999 | 0.0000 |

**解释**：SGTONetV6 不是准确率最高的模型，iTransformer 的 accuracy 更高。可辩护的 claim 应该是：SGTONetV6 改善了 macro-level fault-state 指标，并且在测试的短时域协议下唯一恢复了稀有 class-9 状态。

### Experiment 2: 消融研究（Figure 3, Table 2）

该实验验证稀有状态恢复是否来自所提出的约束触发机制。

| Variant | Macro-F1 | Balanced Acc. | Class9 F1 |
|---|---:|---:|---:|
| **Full SGTONetV6DualOverride** | **0.6233** | **0.6731** | **0.5556** |
| No precursor constraint | 0.6139 | 0.6725 | 0.5101 |
| Mean rare context | 0.5848 | 0.6654 | 0.2317 |
| No fallback prior | 0.5830 | 0.6397 | 0.3556 |
| No rare override | 0.5113 | 0.5568 | 0.0000 |
| No boundary constraint | 0.4550 | 0.5469 | 0.0158 |

**解释**：rare override 是必要的，因为仅靠基础分类器无法恢复 class `9`。boundary constraint 也必要，因为无约束触发会产生失控的稀有类别预测。patch-attentive rare context 强于简单 mean context，说明稀有退化证据可能只出现在输入窗口的局部片段中。

### Experiment 3: Rare-trigger 校准与阈值敏感性（Figure 6）

SGTONetV6 在可行时使用验证集进行阈值校准。由于 class `9` 极度稀疏，一些验证划分中稀有样本太少，无法稳定校准。因此实现中加入 fallback threshold prior。

完整模型的 mean rare override threshold 约为 `0.0097`。在保存的 threshold-sensitivity 曲线中，最佳全局测试阈值约为 `0.009`，对应 macro-F1 `0.6069`、class-9 F1 `0.4732`。

**解释**：阈值校准会影响 precision-recall tradeoff。主结果应基于三划分校准协议，阈值曲线应作为 sensitivity analysis 呈现。

### Experiment 4: 混淆矩阵分析（Figure 5）

使用已保存的 confusion matrix figure 对比 SGTONetV6DualOverride 和 iTransformer。

**预期信息**：iTransformer 通过较好建模主导类别获得高总体准确率，但漏检 class `9`。SGTONetV6 在主导类别准确率上有一定牺牲，但恢复了稀有二级退化状态。

### Experiment 5: Horizon transfer 局限性（Figure 4）

该实验测试同一 SGTONetV6 设计是否能从 `label_shift=1` 直接迁移到 `label_shift=3`。

| Horizon | Model | Macro-F1 | Balanced Acc. | Class9 Precision | Class9 Recall | Class9 F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | SGTONetV6DualOverride | 0.6233 | 0.6731 | 0.5611 | 0.5833 | 0.5556 |
| 1 | PatchTST | 0.5687 | 0.6154 | 0.0000 | 0.0000 | 0.0000 |
| 3 | PatchTST | 0.5877 | 0.6310 | 0.1789 | 0.2597 | 0.1919 |
| 3 | SGTONetV6DualOverride | 0.5006 | 0.6112 | 0.0620 | 0.4762 | 0.1070 |
| 3 | SGTONetV4Conservative | 0.4805 | 0.5466 | 0.0000 | 0.0000 | 0.0000 |

**解释**：当前 SGTONetV6 claim 应限制在短时域预测。在 `label_shift=3` 下，rare trigger 仍提高了 recall，但 precision 明显下降，false positives 成为主要问题。

## Figures

1. **Figure 1**：来自 `results/sgto_v6_dual/figures/fig1_main_d1_metrics.pdf` 的柱状图或分组指标图。展示不同模型的 accuracy、macro-F1、balanced accuracy 和 fault macro-F1。
2. **Figure 2**：来自 `results/sgto_v6_dual/figures/fig2_class9_prf1.pdf` 的 class-9 precision/recall/F1 图。突出 baseline 的稀有类别崩塌和 SGTONetV6 的恢复能力。
3. **Figure 3**：来自 `results/sgto_v6_dual/figures/fig3_ablation.pdf` 的消融图。展示 rare override、boundary constraint、fallback prior 和 patch-attentive context 的作用。
4. **Figure 4**：来自 `results/sgto_v6_dual/figures/fig4_horizon_transfer.pdf` 的 horizon-transfer 图。用于 discussion 或 appendix 来界定 claim 边界。
5. **Figure 5**：来自 `results/sgto_v6_dual/figures/fig5_confusion_v6_vs_itransformer.pdf` 的 confusion matrix 对比。展示 class-9 collapse 与 recovery。
6. **Figure 6**：来自 `results/sgto_v6_dual/figures/fig6_threshold_sensitivity.pdf` 的 threshold sensitivity 曲线。展示 rare-trigger calibration 行为。
7. **Table 1**：主对比表，来源 `results/sgto_v6_dual/full_d1_main_comparison.csv`。
8. **Table 2**：消融表，来源 `results/sgto_v6_dual/final_main_and_ablations.csv`。

## Known Weaknesses

- 最强证据来自单个 Hoister 私有数据集，目前还没有 public-dataset sanity check。
- 方法没有取得最高 overall accuracy。论文必须把 accuracy 作为次要指标，把 fault macro-F1 和 rare-class recovery 作为核心。
- 相比 iTransformer，macro-F1 提升幅度较小：`0.6233` vs. `0.6185`，因此不能声称广义分类性能全面优越。
- 当前方法不能干净迁移到 `label_shift=3`；不支持 multi-horizon superiority。
- Rare-trigger 阈值校准依赖稀疏验证证据。fallback threshold 有用，但需要诚实解释。
- 投稿前仍需核验引用和 BibTeX。

## Related Work

- **工业时间序列故障诊断**：多变量传感器故障诊断、预测性维护、健康状态分类、工业监测。
- **早期与未来状态时间序列分类**：early classification、lead-time fault prediction、fixed-horizon future-state prediction。
- **类别不均衡与稀有事件检测**：class-weighted loss、focal loss、重采样、异常分数、阈值校准、rare-event precision-recall tradeoff。
- **边界感知时间监督**：transition-sensitive supervision、noisy temporal labels、label-shift windows、boundary-aware inference rules。
- **时间序列 backbone**：DLinear、TimesNet、iTransformer、PatchTST 和 graph-enhanced temporal networks。

不要编造具体参考文献。文献检索后再补充 verified citations。

## Proposed Title

"Boundary-Constrained Rare-Fault Triggering for Short-Horizon Hoister Fault-State Prediction"

备选标题：

"SGTONet: Shift-Aware Boundary Triggering for Rare Fault-State Prediction in Hoisting Systems"

## Target Venue

首选目标：IEEE IAS conference 风格的工业应用论文。

可能的扩展目标：IEEE Transactions on Industry Applications 或其他工业信息学方向期刊，但通常需要更强验证，例如 public-dataset sanity check、多现场数据或更多运行工况。
