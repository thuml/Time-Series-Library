# Paper Plan

> **Workflow 3 模板：跳过 planning phase 后可直接使用。** 本文件针对当前 SGTONetV6 Hoister 未来状态预测故事填写，并保持 `PAPER_PLAN_TEMPLATE.md` 的格式。

## Metadata
- **Title**: Boundary-Constrained Rare-Fault Triggering for Short-Horizon Hoister Fault-State Prediction
- **Venue**: IEEE IAS conference，后续在验证增强后可扩展为 journal version
- **One-sentence contribution**: SGTONetV6 通过把保守多类别预测与边界约束稀有故障触发拆开，在严重稀有状态不均衡下提升短时域 Hoister 未来状态预测。

## Claims-Evidence Matrix
| # | Claim | Evidence | Section |
|---|-------|----------|---------|
| C1 | 固定时域 Hoister 未来状态预测存在 rare-boundary collapse：强 baseline 可以保持高 accuracy，但完全漏检 class `9`。 | iTransformer accuracy `0.8175`，但 class-9 F1 `0.0000`；DLinear、TimesNet、PatchTST 和 SGTONetV4Conservative 的 class-9 F1 也都是 `0.0000`。 | §1, §4 |
| C2 | SGTONetV6 在 `label_shift=1` 下恢复稀有二级退化状态，同时提升 macro-level fault metrics。 | SGTONetV6DualOverride: macro-F1 `0.6233`, balanced accuracy `0.6731`, fault macro-F1 `0.5411`, class-9 F1 `0.5556`。 | §4 |
| C3 | 提升来自 boundary-constrained rare triggering，而不是简单换成更强的 temporal backbone。 | No rare override 的 class-9 F1 为 `0.0000`；no boundary constraint 为 `0.0158`；mean rare context 为 `0.2317`；full model 为 `0.5556`。 | §3, §4 |
| C4 | 当前结果支持短时域 claim，不支持一般 multi-horizon claim。 | 在 `label_shift=3` 下，PatchTST macro-F1 `0.5877`、class-9 F1 `0.1919`，而 SGTONetV6 macro-F1 `0.5006`、class-9 F1 `0.1070`。 | §4, §5 |

## Section Plan

### 1. Introduction (~1.5 pages)
- **What**: 将提升机超速监测表述为固定时域未来状态分类，即根据当前多变量传感器窗口预测下一短时域运行状态。
- **Why**: 安全导向监测不能只看 overall accuracy，因为稀有二级退化状态 class `9` 可能被强分类器完全漏检。
- **How**: 提出 SGTONetV6，一种 dual-mode 模型，包含保守未来状态分类器和由 transition boundary 与 precursor-state 语义约束的校准稀有故障触发器。
- **Result**: 在 27 文件 Hoister 私有数据集的 `label_shift=1` 设置下，SGTONetV6 达到 macro-F1 `0.6233`、fault macro-F1 `0.5411`、class-9 F1 `0.5556`；所有测试的非 trigger baseline 的 class-9 F1 都是 `0.0000`。

Introduction 的关键逻辑：

1. 工业提升系统需要在稀有退化被漏检前进行短时域状态预测。
2. 私有数据集包含五个有序运行/故障状态：`1`、`5`、`7`、`9`、`3`。
3. Class `9` 极度稀疏：37,417 个时间戳中只有 185 个。
4. Baseline accuracy 具有误导性：iTransformer accuracy 最高，但 class-9 F1 为零。
5. 因此论文目标不是泛化的最高 accuracy，而是 rare-boundary recovery。

### 2. Related Work (~1 page)
- **Industrial time-series fault diagnosis**: 讨论基于传感器的故障诊断、预测性维护、健康状态分类。Gap：许多研究强调 aggregate classification，但没有显式控制稀有 transition-state recovery。
- **Early and future-state classification**: 将本文任务定位为 fixed-horizon future-state prediction，而不是任意 early stopping。Gap：状态转移附近的 label shift 会产生 rare-boundary 样本。
- **Class imbalance and rare-event prediction**: 覆盖 class weights、focal loss、resampling、threshold calibration、anomaly detection、rare-event precision-recall tradeoff。Gap：这些方法不直接编码稀有状态何时在物理/工况上合理。
- **Boundary-aware temporal learning**: 讨论 transition-sensitive losses 和 noisy temporal labels。Gap：SGTONetV6 在 rare-trigger inference 阶段使用 boundary semantics。
- **Time-series backbones**: 将 DLinear、TimesNet、iTransformer、PatchTST 和 SGTO variants 作为相关 baseline。

不要加入未经核验的引用。最终 LaTeX 写作前需要文献核查。

### 3. Method (~2 pages)
- **Problem formulation**: 令 `W_t = {x_{t-L+1}, ..., x_t}` 为输入窗口，预测 `y_{t+Delta}`。主论文设置为 `Delta=1`。标签集合为 `{1, 5, 7, 9, 3}`，其中 class `9` 是 rare target。
- **CSV future-state target construction**: loader 使用 `seq_len=96`、`window_step=8`、`label_shift=1` 构造滑动窗口。启用 `enable_future_state_targets` 后，每个样本包含 future label、current label、boundary flag 和 future feature window。
- **Patch temporal encoder**: SGTONetV6 继承 SGTO patch temporal encoder，输出 patch tokens 和 window-level hidden representation。
- **Conservative future-state classifier**: base path 使用 graph-aware transition refinement、destination experts、prototype logits 和 future-state head 预测五个未来状态。
- **Patch-attentive rare context**: rare query 通过 key/value projections 对 patch tokens 做注意力汇聚，形成稀有状态的局部证据。
- **Rare trigger head**: rare branch 拼接 hidden state、future hidden state、rare context、current probabilities、transition prior、rare prototype similarity 和 boundary logit，输出 scalar rare score。
- **Boundary-constrained inference override**: 只有 rare score 超过校准阈值且样本满足 boundary 与 precursor constraints 时，base prediction 才会被改为 class `9`。

核心推理规则：

```text
if rare_score >= tau and boundary_flag and current_label in {5, 7}:
    pred = class9
else:
    pred = base_classifier_prediction
```

重要表述：

- rare trigger 不是对 multiclass classifier 的全局替代。
- base classifier 保持保守，以避免过多 rare false alarms。
- boundary 和 precursor rules 编码了 class `9` 的工况合理性。
- fallback threshold prior 用于处理验证集中稀有样本太少、无法稳定校准的情况。

### 4. Experiments (~3 pages)
- **Dataset**: Hoister 私有数据集，27 个 CSV 文件，37,417 个时间戳，20 列。目标列为 `running_state_five_class`。丢弃 `JianSuDuan_ChaoSu`，因为它是直接故障指示变量，可能泄漏故障状态信息。
- **Class distribution**: label `1`: 10,959；label `5`: 15,695；label `7`: 5,364；label `9`: 185；label `3`: 5,214。
- **Protocol**: 使用文件级划分 seeds `{14, 22, 30}`，`seq_len=96`，`window_step=8`，`label_shift=1`，batch size `16`，class weights，以及 macro-F1-oriented early stopping。
- **Baselines**: DLinear、TimesNet、iTransformer、PatchTST、SGTONetV4Conservative。
- **Main results**: 使用 `results/sgto_v6_dual/full_d1_main_comparison.csv`。强调 macro-F1、balanced accuracy、fault macro-F1 和 class-9 F1。报告 accuracy，但不要把它作为主 claim。
- **Ablation**: 使用 `results/sgto_v6_dual/final_main_and_ablations.csv`。包括 full model、no precursor constraint、mean rare context、no fallback prior、no rare override、no boundary constraint。
- **Threshold sensitivity**: 使用 `results/sgto_v6_dual/threshold_sensitivity_curve.csv` 和 Figure 6。作为 calibration analysis，而不是 headline result。
- **Horizon limitation**: 使用 `results/sgto_v6_dual/horizon1_vs_horizon3_summary.csv`。明确 `label_shift=3` 不支持 general multi-horizon superiority。

主结果表：

| Model | Accuracy | Macro-F1 | Balanced Acc. | Fault Macro-F1 | Class9 F1 |
|---|---:|---:|---:|---:|---:|
| SGTONetV6DualOverride | 0.7102 | 0.6233 | 0.6731 | 0.5411 | 0.5556 |
| iTransformer | 0.8175 | 0.6185 | 0.6594 | 0.4615 | 0.0000 |
| DLinear | 0.7895 | 0.5961 | 0.6466 | 0.4426 | 0.0000 |
| TimesNet | 0.7958 | 0.5893 | 0.6207 | 0.4275 | 0.0000 |
| PatchTST | 0.7559 | 0.5687 | 0.6154 | 0.4194 | 0.0000 |
| SGTONetV4Conservative | 0.7630 | 0.5605 | 0.5961 | 0.3999 | 0.0000 |

### 5. Conclusion (~0.5 pages)
- **Summary**: SGTONetV6 通过把保守未来状态分类与边界约束稀有触发拆开，解决短时域 Hoister 未来状态预测中的 rare-boundary collapse。
- **Limitations**: 证据来自单个私有数据集；方法 overall accuracy 低于 iTransformer；当前 rare-trigger calibration 不能干净迁移到 `label_shift=3`。
- **Future**: 增加 public 或 multi-site validation，改进 horizon-specific calibration，并评估部署导向的 false-alarm cost。

## Figure Plan
| # | Type | Description | Auto? |
|---|------|-------------|:-----:|
| Fig 1 | Architecture | SGTONetV6 overview: patch encoder, conservative classifier, rare context, rare trigger, boundary-constrained override | illustration |
| Fig 2 | Bar chart | Main `label_shift=1` metric comparison from `fig1_main_d1_metrics.pdf` | matplotlib |
| Fig 3 | Bar chart | Class-9 precision/recall/F1 from `fig2_class9_prf1.pdf` | matplotlib |
| Fig 4 | Bar chart | Ablation study from `fig3_ablation.pdf` | matplotlib |
| Fig 5 | Confusion matrix | SGTONetV6 versus iTransformer from `fig5_confusion_v6_vs_itransformer.pdf` | matplotlib |
| Fig 6 | Line plot | Rare-trigger threshold sensitivity from `fig6_threshold_sensitivity.pdf` | matplotlib |
| Appendix Fig | Line or grouped chart | Horizon-1 versus horizon-3 transfer limitation from `fig4_horizon_transfer.pdf` | matplotlib |
| Table 1 | Comparison | Main model comparison from `full_d1_main_comparison.csv` | LaTeX |
| Table 2 | Ablation | Mechanism ablations from `final_main_and_ablations.csv` | LaTeX |

## Key References
1. DLinear：最终写作前核验官方论文引用。
2. TimesNet：最终写作前核验官方论文引用。
3. iTransformer：最终写作前核验官方论文引用。
4. PatchTST：最终写作前核验官方论文引用。
5. Industrial fault diagnosis and predictive maintenance references：文献检索后补充。
6. Class imbalance, focal loss, and rare-event detection references：文献检索后补充。
7. Early time-series classification and lead-time fault prediction references：文献检索后补充。
