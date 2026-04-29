# SGTONetV6 Paper Plan

Date: 2026-04-25

## Current Decision

The previous manuscript package is written around `SGPH-Net`. The current experimental evidence supports a different and narrower paper line:

> SGTONetV6 improves fixed short-horizon hoister future-state classification under severe rare-fault imbalance by decoupling conservative future-state classification from boundary-constrained rare-fault triggering.

This should become the main paper story. Do not continue the old `SGPH-Net` hazard/time-bucket story unless new warning experiments are run.

## Paper Type

Recommended type: application-driven industrial time-series method paper.

Recommended venue framing:

- IEEE IAS conference: feasible if positioned as an industrial fault-state prediction study.
- IEEE TIA-style journal extension: needs stronger validation, preferably public-data sanity check or additional site/device data.
- Generic ML conference: currently too narrow and private-data-dependent.

## Claims-Evidence Matrix

| Claim | Evidence | Status | Paper Location |
|---|---|---|---|
| Fixed-horizon future-state classification has a rare-boundary failure mode: strong baselines can achieve high accuracy while missing rare class 9. | iTransformer accuracy `0.8175` but class9 F1 `0.0000`; all listed baselines have class9 F1 `0.0000`. | Supported on private Hoister data | Introduction, Experiments |
| SGTONetV6 recovers rare class 9 while preserving competitive macro-level performance. | SGTONetV6 macro-F1 `0.6233`, fault macro-F1 `0.5411`, class9 F1 `0.5556`. | Supported for `label_shift=1` | Main Results |
| The gain comes from boundary-constrained rare triggering, not just a stronger backbone. | No rare override gives class9 F1 `0.0000`; no boundary constraint gives class9 F1 `0.0158`; mean rare context gives class9 F1 `0.2317`. | Supported by ablations | Ablation |
| The current method is not a general multi-horizon solution. | At `label_shift=3`, PatchTST macro-F1 `0.5877` vs SGTONetV6 `0.5006`. | Negative evidence, should be disclosed as limitation | Discussion or Appendix |

## Recommended Title

Primary:

> Boundary-Constrained Rare-Fault Triggering for Short-Horizon Hoister Fault-State Prediction

Alternative:

> SGTONet: Shift-Aware Boundary Triggering for Rare Fault-State Prediction in Hoisting Systems

Avoid titles that promise broad multi-horizon forecasting or generic time-series classification superiority.

## Section Plan

### Abstract

Key message:

- Problem: future-state classification for an industrial hoister fault process is dominated by rare transition states.
- Gap: high-accuracy classifiers can miss the rare second-level degradation state entirely.
- Method: SGTONetV6 combines a conservative future-state classifier with a calibrated rare-fault trigger constrained by transition boundaries and precursor states.
- Result: On `label_shift=1`, SGTONetV6 achieves macro-F1 `0.6233` and class9 F1 `0.5556`, while DLinear, TimesNet, iTransformer, PatchTST, and conservative SGTONetV4 all have class9 F1 `0.0000`.
- Limitation: gains are strongest for short-horizon prediction and should not be claimed as multi-horizon generalization.

### 1. Introduction

Opening:

Industrial hoisting faults are not only detected after occurrence; operators need short-horizon state prediction before a rare but critical degradation stage is missed.

Problem gap:

Most time-series classifiers optimize average accuracy or macro metrics, but in this dataset the critical failure is class-specific: rare class 9 can disappear from predictions while overall accuracy remains high.

Contributions:

1. Formalize fixed-horizon hoister future-state classification with `label_shift=1`.
2. Identify rare-boundary collapse as the key failure mode of standard classifiers.
3. Propose SGTONetV6, a dual-mode model with conservative classification and boundary-constrained rare triggering.
4. Provide ablation and sensitivity evidence showing that rare override, boundary constraint, fallback prior, and patch-attentive rare context are necessary.

Hero figure:

- Left: standard classifier predicts dominant states and misses class 9.
- Right: SGTONetV6 keeps conservative base predictions but activates a rare trigger only near valid transition boundaries.
- Include a small metric callout: class9 F1 `0.0000 -> 0.5556`.

### 2. Related Work

Subtopics:

- Industrial time-series fault diagnosis and predictive maintenance.
- Early time-series classification and lead-time prediction.
- Class imbalance and rare-event detection.
- Boundary/noisy-label supervision in temporal classification.

Positioning:

The paper is not another generic backbone. It targets fixed-horizon future-state classification where temporal label shift and rare transition states create a specific failure mode.

### 3. Problem Formulation

Define:

- Input window: `x[t-L+1:t]`
- Current label: `y_t`
- Future label: `y_{t+Delta}`
- Main task: predict `y_{t+1}` from the current window
- State set: `{1, 5, 7, 9, 3}`
- Boundary flag: whether current and future labels cross a transition boundary
- Rare target: class `9`

Make clear that `label_shift=1` is the main paper setting.

### 4. Method: SGTONetV6

Recommended method decomposition:

1. Patch temporal encoder.
2. Conservative future-state classifier.
3. Patch-attentive rare context module.
4. Rare trigger head.
5. Boundary and precursor constrained inference override.
6. Calibration/fallback rule for rare threshold.

Core inference rule:

```text
if rare_score >= tau and boundary_flag and current_label in {5, 7}:
    pred = class9
else:
    pred = base_classifier_prediction
```

Important wording:

- The base classifier is intentionally conservative.
- The rare trigger is not allowed to fire everywhere; it is constrained by boundary and precursor semantics.
- This is the main difference from simply using class weights or focal loss.

### 5. Experiments

Main table:

Use `results/sgto_v6_dual/full_d1_main_comparison.csv`.

Required columns:

- Accuracy
- Macro-F1
- Balanced accuracy
- Fault macro-F1
- Class9 precision/recall/F1

Main baselines:

- DLinear
- TimesNet
- iTransformer
- PatchTST
- SGTONetV4Conservative

Main figures:

- `fig1_main_d1_metrics.pdf`
- `fig2_class9_prf1.pdf`
- `fig5_confusion_v6_vs_itransformer.pdf`

### 6. Ablation and Analysis

Use `results/sgto_v6_dual/final_main_and_ablations.csv`.

Required ablations:

- Full SGTONetV6DualOverride
- No precursor constraint
- Mean context instead of patch-attentive context
- No fallback prior
- No rare override
- No boundary constraint

Key statement:

Removing rare override collapses class9 F1 to zero; removing boundary constraint causes rare false triggers and class9 F1 falls to `0.0158`.

Include:

- `fig3_ablation.pdf`
- `fig6_threshold_sensitivity.pdf`

### 7. Discussion and Limitations

Must state:

- The method improves rare-fault recovery at `label_shift=1`.
- Accuracy is lower than iTransformer, so deployment should choose metrics based on safety objective.
- `label_shift=3` does not support a multi-horizon superiority claim.
- Results are from one private dataset; public or multi-site validation is future work.

Use `fig4_horizon_transfer.pdf` either here or in appendix to honestly bound the claim.

## Figure and Table Plan

| ID | Source | Purpose | Priority |
|---|---|---|---|
| Fig. 1 | New manual architecture figure | Explain conservative classifier + boundary rare trigger | High |
| Fig. 2 | `fig1_main_d1_metrics.pdf` | Main metric comparison | High |
| Fig. 3 | `fig2_class9_prf1.pdf` | Rare class recovery | High |
| Fig. 4 | `fig3_ablation.pdf` | Mechanism evidence | High |
| Fig. 5 | `fig5_confusion_v6_vs_itransformer.pdf` | Show class9 collapse vs recovery | Medium |
| Fig. 6 | `fig6_threshold_sensitivity.pdf` | Threshold calibration sensitivity | Medium |
| Appendix Fig. | `fig4_horizon_transfer.pdf` | Scope limitation for `Delta=3` | Medium |
| Table 1 | `full_d1_main_comparison.csv` | Main comparison | High |
| Table 2 | `final_main_and_ablations.csv` | Ablation table | High |

## Required Edits to Existing Draft

The current files under `article/hoister_ias2026/en/` should be rewritten rather than lightly patched:

- Replace `SGPH-Net` with `SGTONetV6` or `SGTONet`.
- Remove hazard head and time-to-fault bucket claims unless new experiments are run.
- Replace current-state recognition with future-state classification.
- Replace empty result tables with the completed `label_shift=1` result tables.
- Add limitations around `label_shift=3`.

## Final Claim Boundary

Safe:

> SGTONetV6 improves short-horizon hoister future-state classification under severe rare-fault imbalance, mainly by recovering the rare second-level degradation class that standard time-series classifiers miss.

Not safe:

- SGTONetV6 is generally better than all baselines.
- SGTONetV6 has the best accuracy.
- SGTONetV6 solves multi-horizon prediction.
- SGTONetV6 generalizes to public datasets without further experiments.
