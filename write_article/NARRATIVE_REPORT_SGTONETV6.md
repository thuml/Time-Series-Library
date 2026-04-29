# Narrative Report: Boundary-Constrained Rare-Fault Triggering for Short-Horizon Hoister Fault-State Prediction

> **This is a SGTONetV6-specific NARRATIVE_REPORT.md for Workflow 3.** It follows the structure of `NARRATIVE_REPORT_EXAMPLE.md`, but replaces the sample long-context Transformer story with the current Hoister private-dataset evidence and the implemented SGTONetV6 line.

## Core Story

We study fixed-horizon future-state prediction for an industrial hoisting overspeed process. The task is not ordinary retrospective fault classification: given a current multivariate sensor window, the model must predict the next short-horizon operating state. The practical difficulty is severe rare-state imbalance around transition boundaries. In the private Hoister dataset, the second-level degradation state, label `9`, appears only 185 times among 37,417 timestamps, yet it is safety-relevant because it occurs before or near a fault process.

Strong time-series classifiers can achieve high overall accuracy while missing this rare state completely. In the current `label_shift=1` experiments, iTransformer reaches accuracy `0.8175`, but its class-9 F1 is `0.0000`. DLinear, TimesNet, PatchTST, and a conservative SGTONetV4 baseline also obtain class-9 F1 `0.0000`.

We propose **SGTONetV6**, a shift-aware graph and trigger oriented network that separates two behaviors: conservative multiclass future-state prediction for common states, and a boundary-constrained rare-fault trigger for class `9`. The implemented model uses a patch temporal encoder inherited from the SGTO line, graph-aware future-state refinement, a prototype-assisted future classifier, a patch-attentive rare context module, and an inference-time rare override rule constrained by boundary and precursor semantics. SGTONetV6 improves macro-F1 and fault macro-F1 under the short-horizon setting and, most importantly, recovers the rare class-9 state.

## Claims

1. **Short-horizon Hoister future-state prediction has a rare-boundary collapse failure mode**: Standard classifiers can score well on aggregate metrics while never predicting the rare second-level degradation state. On the private Hoister task with `label_shift=1`, all tested non-trigger baselines have class-9 F1 `0.0000`.

2. **SGTONetV6 recovers rare class `9` by decoupling the base classifier from the rare trigger**: The full SGTONetV6DualOverride model reaches macro-F1 `0.6233`, balanced accuracy `0.6731`, fault macro-F1 `0.5411`, and class-9 F1 `0.5556`, while the strongest baseline macro-F1 is iTransformer at `0.6185` with class-9 F1 `0.0000`.

3. **The mechanism is boundary-constrained rather than a generic stronger backbone**: Removing the rare override reduces class-9 F1 to `0.0000`; removing the boundary constraint reduces class-9 F1 to `0.0158`; replacing patch-attentive rare context with mean context reduces class-9 F1 to `0.2317`.

4. **The current evidence supports a scoped short-horizon claim, not a broad multi-horizon claim**: At `label_shift=3`, PatchTST obtains macro-F1 `0.5877` and class-9 F1 `0.1919`, while SGTONetV6DualOverride obtains macro-F1 `0.5006` and class-9 F1 `0.1070`. This should be disclosed as a limitation.

## Experiments

### Setup
- **Model**: SGTONetV6DualOverride, implemented in `models/SGTONetV6.py`
- **Data**: Private Hoister overspeed dataset, 27 CSV files, 37,417 timestamps, 20 columns
- **Target**: `running_state_five_class`
- **Dropped columns**: `id`, `time`, `JianSuDuan_ChaoSu`, `running_state_class`, `running_state_five_class`
- **States**: label `1` stop, label `5` normal, label `7` first-level degradation, label `9` second-level degradation, label `3` fault
- **Windowing**: `seq_len=96`, `window_step=8`, `window_label_mode=last`
- **Prediction horizon**: main evidence uses `label_shift=1`
- **Splits**: file-level split seeds `14`, `22`, and `30`
- **Baselines**: DLinear, TimesNet, iTransformer, PatchTST, SGTONetV4Conservative
- **Metrics**: accuracy, macro-F1, weighted F1, balanced accuracy, fault macro-F1, class-9 precision/recall/F1

### Dataset Summary

The dataset is stored under `dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579`. It contains 27 CSV files and 37,417 rows. Each CSV has the following 20 columns:

```text
id, time, SuDuMoNiLiang, FPLCSuDu, BianmaQiSuDu, CSJSuDu,
LGuanLongShenDu, FPLCShenDu, BianmaQiShenDu, DianshuDianliu1,
DianshuDianliu2, LiCi_Current, ZhiDongPressure, ZhuLingDWQ1,
ZhaBaDWQ1, WZhuJiLiang, WFuJiLiang, JianSuDuan_ChaoSu,
running_state_class, running_state_five_class
```

The verified five-class label distribution is:

| Label | Meaning | Count |
|---:|---|---:|
| 1 | Stop | 10,959 |
| 5 | Normal operation | 15,695 |
| 7 | First-level degradation | 5,364 |
| 9 | Second-level degradation | 185 |
| 3 | Fault occurrence | 5,214 |

**Interpretation**: Class `9` accounts for less than 0.5% of timestamps. This makes class-9 recovery the central safety-oriented evaluation problem.

### Experiment 1: Main Short-Horizon Comparison (Table 1, Figures 1-2)

Compared SGTONetV6DualOverride against DLinear, TimesNet, iTransformer, PatchTST, and SGTONetV4Conservative on the `label_shift=1` future-state task.

| Model | Accuracy | Macro-F1 | Balanced Acc. | Fault Macro-F1 | Class9 F1 |
|---|---:|---:|---:|---:|---:|
| **SGTONetV6DualOverride** | 0.7102 | **0.6233** | **0.6731** | **0.5411** | **0.5556** |
| iTransformer | **0.8175** | 0.6185 | 0.6594 | 0.4615 | 0.0000 |
| DLinear | 0.7895 | 0.5961 | 0.6466 | 0.4426 | 0.0000 |
| TimesNet | 0.7958 | 0.5893 | 0.6207 | 0.4275 | 0.0000 |
| PatchTST | 0.7559 | 0.5687 | 0.6154 | 0.4194 | 0.0000 |
| SGTONetV4Conservative | 0.7630 | 0.5605 | 0.5961 | 0.3999 | 0.0000 |

**Interpretation**: SGTONetV6 is not the highest-accuracy model; iTransformer has higher accuracy. The defensible claim is that SGTONetV6 improves macro-level fault-state metrics and uniquely recovers the rare class-9 state under the tested short-horizon protocol.

### Experiment 2: Ablation Study (Figure 3, Table 2)

Tested whether rare recovery comes from the proposed constrained trigger mechanism.

| Variant | Macro-F1 | Balanced Acc. | Class9 F1 |
|---|---:|---:|---:|
| **Full SGTONetV6DualOverride** | **0.6233** | **0.6731** | **0.5556** |
| No precursor constraint | 0.6139 | 0.6725 | 0.5101 |
| Mean rare context | 0.5848 | 0.6654 | 0.2317 |
| No fallback prior | 0.5830 | 0.6397 | 0.3556 |
| No rare override | 0.5113 | 0.5568 | 0.0000 |
| No boundary constraint | 0.4550 | 0.5469 | 0.0158 |

**Interpretation**: The rare override is necessary because the base classifier alone does not recover class `9`. The boundary constraint is also necessary because an unconstrained trigger produces uncontrolled rare predictions. Patch-attentive rare context is stronger than simple mean context, supporting the hypothesis that rare evidence is localized inside the input window.

### Experiment 3: Rare-Trigger Calibration and Threshold Sensitivity (Figure 6)

SGTONetV6 uses validation-based threshold calibration when possible. Because class `9` is extremely sparse, some validation splits contain too few rare samples for stable calibration. The implementation therefore allows a fallback threshold prior.

The full model uses a mean rare override threshold of approximately `0.0097`. In the saved threshold-sensitivity curve, the best tested global threshold is around `0.009`, with macro-F1 `0.6069` and class-9 F1 `0.4732`.

**Interpretation**: Threshold calibration affects the precision-recall tradeoff. The main result should rely on the three-split calibrated protocol, while the threshold curve should be presented as sensitivity analysis.

### Experiment 4: Confusion Matrix Analysis (Figure 5)

Compare SGTONetV6DualOverride and iTransformer using the saved confusion matrix figure.

**Expected message**: iTransformer obtains high aggregate accuracy by modeling dominant states well, but it misses class `9`. SGTONetV6 sacrifices some accuracy on dominant classes to recover the rare second-level degradation state.

### Experiment 5: Horizon Transfer Limitation (Figure 4)

Tested whether the same SGTONetV6 design transfers directly from `label_shift=1` to `label_shift=3`.

| Horizon | Model | Macro-F1 | Balanced Acc. | Class9 Precision | Class9 Recall | Class9 F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | SGTONetV6DualOverride | 0.6233 | 0.6731 | 0.5611 | 0.5833 | 0.5556 |
| 1 | PatchTST | 0.5687 | 0.6154 | 0.0000 | 0.0000 | 0.0000 |
| 3 | PatchTST | 0.5877 | 0.6310 | 0.1789 | 0.2597 | 0.1919 |
| 3 | SGTONetV6DualOverride | 0.5006 | 0.6112 | 0.0620 | 0.4762 | 0.1070 |
| 3 | SGTONetV4Conservative | 0.4805 | 0.5466 | 0.0000 | 0.0000 | 0.0000 |

**Interpretation**: The current SGTONetV6 claim should be restricted to short-horizon prediction. At `label_shift=3`, the rare trigger still improves recall but loses precision, so false positives dominate.

## Figures

1. **Figure 1**: Bar chart or grouped metric plot from `results/sgto_v6_dual/figures/fig1_main_d1_metrics.pdf`. Shows accuracy, macro-F1, balanced accuracy, and fault macro-F1 across models.
2. **Figure 2**: Class-9 precision/recall/F1 plot from `results/sgto_v6_dual/figures/fig2_class9_prf1.pdf`. Highlights rare-state collapse in baselines and recovery by SGTONetV6.
3. **Figure 3**: Ablation plot from `results/sgto_v6_dual/figures/fig3_ablation.pdf`. Shows the role of rare override, boundary constraint, fallback prior, and patch-attentive context.
4. **Figure 4**: Horizon-transfer plot from `results/sgto_v6_dual/figures/fig4_horizon_transfer.pdf`. Used in discussion or appendix to bound the claim.
5. **Figure 5**: Confusion matrix comparison from `results/sgto_v6_dual/figures/fig5_confusion_v6_vs_itransformer.pdf`. Shows class-9 collapse versus recovery.
6. **Figure 6**: Threshold sensitivity curve from `results/sgto_v6_dual/figures/fig6_threshold_sensitivity.pdf`. Shows rare-trigger calibration behavior.
7. **Table 1**: Main comparison table from `results/sgto_v6_dual/full_d1_main_comparison.csv`.
8. **Table 2**: Ablation table from `results/sgto_v6_dual/final_main_and_ablations.csv`.

## Known Weaknesses

- The strongest evidence is from one private Hoister dataset. There is no public-dataset sanity check yet.
- The method does not achieve the highest overall accuracy. The paper must frame accuracy as secondary to fault macro-F1 and rare-class recovery.
- The macro-F1 margin over iTransformer is small (`0.6233` vs. `0.6185`), so the paper should not claim broad classifier superiority.
- The current method does not transfer cleanly to `label_shift=3`; multi-horizon superiority is not supported.
- Rare-trigger threshold calibration depends on sparse validation evidence. The fallback threshold is useful but should be explained honestly.
- Citations and BibTeX entries still need to be verified before submission.

## Related Work

- **Industrial Time-Series Fault Diagnosis**: multivariate sensor-based fault diagnosis, predictive maintenance, health-state classification, and industrial monitoring.
- **Early and Future-State Time-Series Classification**: early classification, lead-time fault prediction, and fixed-horizon future-state prediction.
- **Class Imbalance and Rare-Event Detection**: class-weighted losses, focal loss, resampling, anomaly scores, threshold calibration, and rare-event precision-recall tradeoffs.
- **Boundary-Aware Temporal Supervision**: transition-sensitive supervision, noisy temporal labels, label-shift windows, and boundary-aware inference rules.
- **Time-Series Backbones**: DLinear, TimesNet, iTransformer, PatchTST, and graph-enhanced temporal networks.

Do not fabricate specific references. Add verified citations after literature checking.

## Proposed Title

"Boundary-Constrained Rare-Fault Triggering for Short-Horizon Hoister Fault-State Prediction"

Alternative title:

"SGTONet: Shift-Aware Boundary Triggering for Rare Fault-State Prediction in Hoisting Systems"

## Target Venue

Primary target: IEEE IAS conference-style industrial application paper.

Possible extension target: IEEE Transactions on Industry Applications or another industrial informatics venue, but this likely requires stronger validation, such as public-dataset sanity checks, multi-site data, or additional operating conditions.
