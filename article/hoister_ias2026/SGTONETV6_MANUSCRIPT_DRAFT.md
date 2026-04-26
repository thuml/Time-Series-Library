# Boundary-Constrained Rare-Fault Triggering for Short-Horizon Hoister Fault-State Prediction

This is a working manuscript draft for the current `SGTONetV6` result line. It intentionally replaces the earlier `SGPH-Net` hazard/time-bucket story with the experimentally supported short-horizon future-state classification story.

## Abstract

Industrial hoisting systems require not only post-event fault recognition, but also short-horizon prediction of degradation states before a fault process becomes critical. In the hoister overspeed dataset studied here, the key challenge is not simply average classification accuracy: standard time-series classifiers can achieve high accuracy while completely missing the rare second-level degradation state. We formulate the task as fixed-horizon future-state classification, where a current multivariate sensor window is used to predict the state label one step ahead. To address rare-state collapse around transition boundaries, we propose SGTONetV6, a dual-mode model that decouples conservative future-state classification from boundary-constrained rare-fault triggering. The model uses a patch temporal encoder, a conservative multiclass classifier, a patch-attentive rare context module, and a calibrated rare-trigger override constrained by boundary and precursor-state semantics. On a private hoister dataset with 27 files and 37,417 timestamps, SGTONetV6 achieves the best macro-F1 among the tested short-horizon models and recovers the rare class-9 state with F1 0.5556, while DLinear, TimesNet, iTransformer, PatchTST, and a conservative SGTO baseline all obtain class-9 F1 of 0.0000. Ablations show that the rare trigger, boundary constraint, fallback prior, and patch-attentive rare context are all necessary for reliable rare-state recovery.

## 1. Introduction

Hoisting systems are safety-critical industrial equipment. During an overspeed fault process, operators need more than a binary indication of whether a fault has already occurred. They need to know whether the system is currently normal, entering an early degradation stage, approaching a severe degradation stage, or already in a fault state. This makes fault-state prediction a short-horizon decision problem rather than a conventional after-the-fact classification problem.

The private hoister dataset used in this study contains five structured states: stop, normal operation, first-level degradation, second-level degradation, and fault occurrence. The second-level degradation state is especially rare, with only 185 timestamps out of 37,417. This imbalance creates a practical failure mode: a model can look accurate while never predicting the rare but safety-relevant class. In our experiments, iTransformer reaches the highest overall accuracy among the tested baselines, but its class-9 F1 is 0.0000. The same rare-class collapse occurs for DLinear, TimesNet, PatchTST, and a conservative SGTO baseline.

This observation motivates a different design objective. Instead of building a heavier generic backbone, we target the boundary region where the future label changes before the current window fully resembles the future state. We argue that short-horizon hoister fault-state prediction needs two behaviors at the same time: conservative multiclass classification for common states, and a selective rare trigger that is only allowed to fire under plausible transition conditions.

We propose SGTONetV6, a shift-aware graph and trigger oriented network for short-horizon future-state classification. The model consists of a patch temporal encoder, a conservative future-state classifier, a patch-attentive rare context module, and a rare-trigger head. At inference time, the rare trigger can override the base classifier only when the rare score exceeds a calibrated threshold and the sample satisfies boundary and precursor constraints. This explicitly separates common-state prediction from rare-state recovery.

The contributions of this paper are:

1. We formulate hoister overspeed monitoring as fixed-horizon future-state classification and identify rare-boundary collapse as a key failure mode of standard classifiers.
2. We propose SGTONetV6, which decouples conservative multiclass prediction from boundary-constrained rare-fault triggering.
3. We provide a controlled comparison against DLinear, TimesNet, iTransformer, PatchTST, and SGTO variants over three file-level split seeds.
4. We show through ablations that rare override, boundary constraint, fallback prior, and patch-attentive context are required for class-9 recovery.

## 2. Related Work

### Industrial time-series fault diagnosis

Industrial fault diagnosis commonly treats multivariate sensor streams as time-series classification inputs and predicts either a fault type or a health state. Modern approaches include convolutional models, recurrent models, transformer-style architectures, graph-enhanced models, and hybrid feature-learning pipelines. These methods are effective when the target classes are sufficiently represented, but they often optimize aggregate classification metrics. In the hoister setting studied here, aggregate accuracy is not enough because a rare degradation state can be completely missed.

### Early and lead-time classification

Early time-series classification studies when a model can emit a label before observing the entire sequence. Lead-time fault prediction similarly asks whether a future state or event can be predicted before it occurs. Our task is narrower: given a fixed current window, predict the label at a fixed future offset. This avoids the stopping-policy problem in early classification, but introduces a label-shift problem near transition boundaries.

### Class imbalance and rare-event prediction

Class imbalance is a central issue in industrial fault datasets. Common remedies include class-weighted losses, focal losses, resampling, anomaly scores, and threshold calibration. However, these methods do not directly encode when a rare state is physically plausible. SGTONetV6 adds this missing constraint by allowing the rare trigger to affect predictions only near valid boundaries and precursor states.

### Boundary-aware temporal supervision

Temporal labels in industrial streams are often imperfectly aligned with the sensor evidence, especially around state transitions. Boundary-aware losses and transition-sensitive supervision can reduce this mismatch. Our method follows this direction, but focuses on inference-time rare-state recovery: the model keeps a conservative future-state classifier and adds a constrained trigger for the rare degradation state.

## 3. Problem Formulation

Let `X = {x_t}_{t=1}^T` be a multivariate industrial time series, where `x_t` contains the sensor measurements at time `t`. For each time index `t`, the model receives a past window:

```text
W_t = {x_{t-L+1}, ..., x_t}.
```

The state label belongs to:

```text
S = {1, 5, 7, 9, 3},
```

where `1` denotes stop, `5` denotes normal operation, `7` denotes first-level degradation, `9` denotes second-level degradation, and `3` denotes fault occurrence.

The main task is fixed-horizon future-state classification:

```text
predict y_{t+Delta} from W_t.
```

The current paper focuses on `Delta = 1`. This is the only horizon currently supported by the main positive evidence. Results for `Delta = 3` are used only to define the limitation of the current method.

We define class `9` as the rare target. A boundary flag indicates whether the current and future labels differ across the prediction offset. The rare trigger is constrained by this boundary flag and by precursor states. In the current implementation, valid precursor labels are `{5, 7}`.

## 4. Method

SGTONetV6 is designed around a simple principle: common-state prediction and rare-state triggering should not be forced into the same flat classifier. The model therefore has two coupled paths.

### Patch temporal encoder

The input window is divided into temporal patches and encoded into a shared representation. This gives the model local temporal context while keeping the encoder compact. The same encoder supports both the base future-state classifier and the rare-trigger branch.

### Conservative future-state classifier

The base classifier predicts the five future states with a standard multiclass head. It is optimized to remain conservative under severe imbalance. This is important because an aggressive rare predictor can create many false alarms and reduce trust in deployment.

### Patch-attentive rare context

The rare branch computes a patch-attentive context representation for the rare class. This mechanism is used because rare degradation evidence may appear only in a short part of the input window. Replacing this module with mean context reduces class-9 F1 from 0.5556 to 0.2317, which supports the need for localized rare evidence.

### Boundary-constrained rare trigger

The rare head outputs a scalar rare score. At inference time, the base prediction can be overridden only when all rare-trigger conditions are satisfied:

```text
rare_score >= tau
AND boundary_flag = true
AND current_label in {5, 7}
```

If the conditions are met, the prediction is changed to class `9`. Otherwise, the model keeps the conservative base prediction. This rule is intentionally restrictive: it improves rare-state recall without allowing the rare head to fire globally.

### Threshold calibration and fallback prior

The rare threshold is calibrated on validation data when possible. Because class `9` is extremely sparse, some validation splits may contain too few rare samples for stable calibration. SGTONetV6 therefore uses a fallback prior threshold. Removing this fallback reduces class-9 F1 from 0.5556 to 0.3556.

## 5. Experiments

### Dataset

The private hoister dataset contains 27 CSV files and 37,417 timestamps sampled every 4 seconds. The verified class counts are:

| Label | Meaning | Count |
|---:|---|---:|
| 1 | Stop | 10,959 |
| 5 | Normal | 15,695 |
| 7 | First-level degradation | 5,364 |
| 9 | Second-level degradation | 185 |
| 3 | Fault | 5,214 |

The target column is `running_state_five_class`. The binary indicator `JianSuDuan_ChaoSu` is dropped from the input because it is a direct fault indicator and appears only in class `3`.

### Protocol

All reported main results use:

- `seq_len = 96`
- `window_step = 8`
- `label_shift = 1`
- split seeds `{14, 22, 30}`
- metrics averaged over the three splits

The primary metrics are macro-F1, balanced accuracy, fault macro-F1, and class-9 precision/recall/F1. Accuracy is reported but is not the main objective because the dataset is highly imbalanced.

### Baselines

The comparison includes DLinear, TimesNet, iTransformer, PatchTST, and SGTONetV4Conservative. These baselines represent strong time-series classification backbones and a conservative SGTO variant without the final dual rare-trigger mechanism.

### Main results

| Model | Accuracy | Macro-F1 | Balanced Acc. | Fault Macro-F1 | Class9 F1 |
|---|---:|---:|---:|---:|---:|
| SGTONetV6DualOverride | 0.7102 | 0.6233 | 0.6731 | 0.5411 | 0.5556 |
| iTransformer | 0.8175 | 0.6185 | 0.6594 | 0.4615 | 0.0000 |
| DLinear | 0.7895 | 0.5961 | 0.6466 | 0.4426 | 0.0000 |
| TimesNet | 0.7958 | 0.5893 | 0.6207 | 0.4275 | 0.0000 |
| PatchTST | 0.7559 | 0.5687 | 0.6154 | 0.4194 | 0.0000 |
| SGTONetV4Conservative | 0.7630 | 0.5605 | 0.5961 | 0.3999 | 0.0000 |

SGTONetV6 obtains the highest macro-F1 and fault macro-F1 among the tested models. More importantly, it is the only method that recovers the rare class-9 state. The margin over iTransformer in macro-F1 is small, but the rare-state difference is large: iTransformer reaches higher accuracy but never predicts class `9` correctly.

### Ablation study

| Variant | Macro-F1 | Balanced Acc. | Class9 F1 |
|---|---:|---:|---:|
| Full SGTONetV6DualOverride | 0.6233 | 0.6731 | 0.5556 |
| No precursor constraint | 0.6139 | 0.6725 | 0.5101 |
| Mean rare context | 0.5848 | 0.6654 | 0.2317 |
| No fallback prior | 0.5830 | 0.6397 | 0.3556 |
| No rare override | 0.5113 | 0.5568 | 0.0000 |
| No boundary constraint | 0.4550 | 0.5469 | 0.0158 |

The ablation results support the proposed mechanism. Without rare override, class-9 F1 collapses to zero. Without the boundary constraint, the rare trigger becomes poorly controlled and class-9 F1 falls to 0.0158. Replacing patch-attentive rare context with mean context also substantially weakens rare-state recovery.

### Threshold sensitivity

A threshold sweep using the constrained inference rule shows the precision-recall tradeoff of the rare trigger. The best tested global threshold in the saved curve is approximately 0.009, with macro-F1 0.6069 and class-9 F1 0.4732. This result should be interpreted as a sensitivity analysis rather than the headline number, because the main experiment uses per-split calibration and fallback.

### Horizon transfer limitation

The current method does not transfer directly to longer prediction horizons. At `label_shift = 3`, PatchTST obtains macro-F1 0.5877 and class-9 F1 0.1919, while SGTONetV6 obtains macro-F1 0.5006 and class-9 F1 0.1070. SGTONetV6 still improves over SGTONetV4Conservative on class-9 recovery, but it does not support a general multi-horizon superiority claim.

## 6. Discussion

The main experimental lesson is that average accuracy is misleading for this task. A model can perform well on dominant states while completely missing the rare second-level degradation state. SGTONetV6 addresses this by separating the stable prediction path from the rare-trigger path. This design makes the method more aligned with safety-oriented industrial monitoring, where rare-state recall can be more important than marginal gains in overall accuracy.

The method is also intentionally scoped. It is not presented as a universal time-series backbone, and it is not currently a general multi-horizon predictor. Its current value is in short-horizon rare-fault recovery under severe class imbalance and boundary-sensitive label shift.

## 7. Conclusion

This paper studies short-horizon hoister future-state classification under severe rare-state imbalance. We show that strong time-series classifiers can achieve high accuracy while missing the rare class-9 degradation state. SGTONetV6 addresses this failure mode by combining conservative multiclass classification with a boundary-constrained rare-fault trigger. On the private Hoister dataset, the method recovers class `9` with F1 0.5556 and improves fault macro-F1 over the tested baselines. Future work should validate the method on public or multi-site industrial datasets and redesign the trigger calibration for longer prediction horizons.

## Figure Placement

Use the generated figures as follows:

| Figure | File | Placement |
|---|---|---|
| Main metrics | `results/sgto_v6_dual/figures/fig1_main_d1_metrics.pdf` | Experiments, main results |
| Class-9 metrics | `results/sgto_v6_dual/figures/fig2_class9_prf1.pdf` | Experiments, main results |
| Ablation | `results/sgto_v6_dual/figures/fig3_ablation.pdf` | Ablation study |
| Confusion matrix | `results/sgto_v6_dual/figures/fig5_confusion_v6_vs_itransformer.pdf` | Error analysis |
| Threshold sensitivity | `results/sgto_v6_dual/figures/fig6_threshold_sensitivity.pdf` | Calibration analysis |
| Horizon transfer | `results/sgto_v6_dual/figures/fig4_horizon_transfer.pdf` | Discussion or appendix |

## Missing Before Submission

1. Add verified citations and BibTeX entries. Do not fabricate references.
2. Draw a clean architecture figure for SGTONetV6.
3. Decide whether to add a public-dataset sanity check. This is the biggest missing item for a stronger paper.
4. Convert this Markdown draft into IEEE LaTeX after the story is accepted.
