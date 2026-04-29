# Paper Plan

> **Template for Workflow 3 — skip planning phase.** This plan is filled for the current SGTONetV6 Hoister future-state prediction story and follows the format of `PAPER_PLAN_TEMPLATE.md`.

## Metadata
- **Title**: Boundary-Constrained Rare-Fault Triggering for Short-Horizon Hoister Fault-State Prediction
- **Venue**: IEEE IAS Annual Meeting 2026 digest, with possible journal extension after stronger validation
- **One-sentence contribution**: SGTONetV6 improves short-horizon Hoister future-state prediction under severe rare-state imbalance by separating conservative multiclass prediction from a boundary-constrained rare-fault trigger.

## Workflow 3 Execution Request

Use this file as the `paper-writing` input plan and skip any generic planning phase unless the template constraints below require compression.

- **Project directory**: `/root/zm/Time-Series-Library-meter-fault_classification_prediction`
- **Writing workspace**: `/root/zm/Time-Series-Library-meter-fault_classification_prediction/write_article`
- **Conference template**: `/root/zm/Time-Series-Library-meter-fault_classification_prediction/article/IAS/IAS_AM2026_Digest_Template_word.docx`
- **Target venue**: IEEE IAS Annual Meeting 2026 digest
- **Anonymous submission**: `true`
- **Page limit**: 5 pages, including key references only
- **Illustration mode**: `false`
- **Reviewer mode**: Codex + Codex; the main Codex agent writes/edits, and `spawn_agent` acts as the reviewer during review and improvement rounds.

Template and formatting requirements:

- Use the provided Word template as the authoritative formatting reference. Do not replace it with a generic IEEE LaTeX template unless a LaTeX intermediate is needed for drafting.
- Final digest output should preserve the IAS AM 2026 Word-template constraints: single column, Times New Roman, 12 pt body text, 1.20 line spacing, anonymous title page, and no author names or affiliations.
- Abstract must be no more than 150 words and should avoid symbols, special characters, footnotes, and math.
- PDF metadata must not expose author details.
- Because this is a 5-page digest, compress the current section plan aggressively: focus on the rare-boundary failure mode, SGTONetV6 mechanism, main short-horizon results, one compact ablation table/figure, and the `label_shift=3` limitation.
- Include only key references in the digest. Keep the complete bibliography candidates separately if needed for a future full paper.

Execution requirements:

- Do not call Gemini and do not require `GEMINI_API_KEY`.
- Do not generate AI bitmap illustrations. If an architecture figure is needed, use a compact manually specified diagram, matplotlib, Mermaid, TikZ, or an existing figure.
- Do not invent experiments, datasets, citations, or numerical results. If evidence is missing, state the gap and keep the claim conservative.
- Prefer existing result files under `results/sgto_v6_dual/` and existing figures under `write_article/figures/` or `results/sgto_v6_dual/figures/`.
- Generate or update a digest manuscript in the writing workspace, then compile/export a PDF if the available toolchain supports the chosen format.
- Run two Codex reviewer/improvement rounds, then report the final manuscript path, PDF path if available, unresolved evidence gaps, and formatting risks.

## Claims-Evidence Matrix
| # | Claim | Evidence | Section |
|---|-------|----------|---------|
| C1 | Fixed-horizon Hoister future-state prediction has a rare-boundary collapse failure mode: strong baselines can miss class `9` entirely while keeping high accuracy. | iTransformer accuracy `0.8175` but class-9 F1 `0.0000`; DLinear, TimesNet, PatchTST, and SGTONetV4Conservative also have class-9 F1 `0.0000`. | §1, §4 |
| C2 | SGTONetV6 recovers the rare second-level degradation state while improving macro-level fault metrics under `label_shift=1`. | SGTONetV6DualOverride: macro-F1 `0.6233`, balanced accuracy `0.6731`, fault macro-F1 `0.5411`, class-9 F1 `0.5556`. | §4 |
| C3 | The improvement comes from boundary-constrained rare triggering rather than simply using a stronger temporal backbone. | No rare override gives class-9 F1 `0.0000`; no boundary constraint gives `0.0158`; mean rare context gives `0.2317`; full model gives `0.5556`. | §3, §4 |
| C4 | The current result supports a short-horizon claim, not a general multi-horizon claim. | At `label_shift=3`, PatchTST macro-F1 `0.5877` and class-9 F1 `0.1919`, while SGTONetV6 macro-F1 `0.5006` and class-9 F1 `0.1070`. | §4, §5 |

## Section Plan

### 1. Introduction (~1.5 pages)
- **What**: Formulate Hoister overspeed monitoring as fixed-horizon future-state classification, where a current multivariate sensor window predicts the next short-horizon operating state.
- **Why**: Safety-oriented monitoring cannot rely only on overall accuracy because the rare second-level degradation state, class `9`, can be completely missed by strong classifiers.
- **How**: Introduce SGTONetV6, a dual-mode model with a conservative future-state classifier and a calibrated rare-fault trigger constrained by transition boundary and precursor-state semantics.
- **Result**: On the private 27-file Hoister dataset with `label_shift=1`, SGTONetV6 reaches macro-F1 `0.6233`, fault macro-F1 `0.5411`, and class-9 F1 `0.5556`; all tested non-trigger baselines have class-9 F1 `0.0000`.

Key introduction flow:

1. Industrial hoisting systems need short-horizon state prediction before rare degradation is missed.
2. The private dataset has five ordered operating/fault states: `1`, `5`, `7`, `9`, and `3`.
3. Class `9` is extremely sparse: 185 timestamps out of 37,417.
4. Baseline accuracy is misleading: iTransformer has the highest accuracy but zero class-9 F1.
5. The paper therefore targets rare-boundary recovery rather than generic top-line accuracy.

### 2. Related Work (~1 page)
- **Industrial time-series fault diagnosis**: Discuss sensor-based fault diagnosis, predictive maintenance, and health-state classification. Gap: many studies emphasize aggregate classification but do not explicitly control rare transition-state recovery.
- **Early and future-state classification**: Position the task as fixed-horizon future-state prediction rather than arbitrary early stopping. Gap: label shift near state transitions creates rare-boundary samples.
- **Class imbalance and rare-event prediction**: Cover class weights, focal loss, resampling, threshold calibration, anomaly detection, and rare-event precision-recall tradeoffs. Gap: these methods do not directly encode when a rare state is physically plausible.
- **Boundary-aware temporal learning**: Discuss transition-sensitive losses and noisy temporal labels. Gap: SGTONetV6 uses boundary semantics during rare-trigger inference.
- **Time-series backbones**: Situate DLinear, TimesNet, iTransformer, PatchTST, and SGTO variants as relevant baselines.

Do not add unverified citations. Literature should be checked before final LaTeX writing.

### 3. Method (~2 pages)
- **Problem formulation**: Let `W_t = {x_{t-L+1}, ..., x_t}` be the input window and predict `y_{t+Delta}`. The main paper setting is `Delta=1`. The label set is `{1, 5, 7, 9, 3}`, with class `9` as the rare target.
- **CSV future-state target construction**: The loader creates sliding windows with `seq_len=96`, `window_step=8`, and `label_shift=1`. When `enable_future_state_targets` is active, each sample includes future label, current label, boundary flag, and future feature window.
- **Patch temporal encoder**: SGTONetV6 inherits the SGTO patch temporal encoder and produces patch tokens plus a window-level hidden representation.
- **Conservative future-state classifier**: The base path predicts the five future states using graph-aware transition refinement, destination experts, prototype logits, and a future-state head.
- **Patch-attentive rare context**: A rare query attends over patch tokens through key/value projections, producing localized rare evidence for the rare trigger.
- **Rare trigger head**: The rare branch combines hidden state, future hidden state, rare context, current probabilities, transition prior, rare prototype similarity, and boundary logit to produce a scalar rare score.
- **Boundary-constrained inference override**: The base prediction is changed to class `9` only when the rare score passes the calibrated threshold and the sample satisfies boundary and precursor constraints.

Core inference rule:

```text
if rare_score >= tau and boundary_flag and current_label in {5, 7}:
    pred = class9
else:
    pred = base_classifier_prediction
```

Important method wording:

- The rare trigger is intentionally not a global replacement for the multiclass classifier.
- The base classifier remains conservative to avoid excessive rare false alarms.
- The boundary and precursor rules encode operational plausibility for class `9`.
- The fallback threshold prior handles validation splits where rare samples are too sparse for stable calibration.

### 4. Experiments (~3 pages)
- **Dataset**: Private Hoister dataset with 27 CSV files, 37,417 timestamps, and 20 columns. Target column is `running_state_five_class`. Drop `JianSuDuan_ChaoSu` because it is a direct fault indicator and would leak fault-state information.
- **Class distribution**: label `1`: 10,959; label `5`: 15,695; label `7`: 5,364; label `9`: 185; label `3`: 5,214.
- **Protocol**: Use file-level splits with seeds `{14, 22, 30}`, `seq_len=96`, `window_step=8`, `label_shift=1`, batch size `16`, class weights, and macro-F1-oriented early stopping.
- **Baselines**: DLinear, TimesNet, iTransformer, PatchTST, and SGTONetV4Conservative.
- **Main results**: Use `results/sgto_v6_dual/full_d1_main_comparison.csv`. Emphasize macro-F1, balanced accuracy, fault macro-F1, and class-9 F1. Report accuracy but do not make it the main claim.
- **Ablation**: Use `results/sgto_v6_dual/final_main_and_ablations.csv`. Include full model, no precursor constraint, mean rare context, no fallback prior, no rare override, no boundary constraint.
- **Threshold sensitivity**: Use `results/sgto_v6_dual/threshold_sensitivity_curve.csv` and Figure 6. Present as calibration analysis, not the headline result.
- **Horizon limitation**: Use `results/sgto_v6_dual/horizon1_vs_horizon3_summary.csv`. State that `label_shift=3` does not support general multi-horizon superiority.

Main result table:

| Model | Accuracy | Macro-F1 | Balanced Acc. | Fault Macro-F1 | Class9 F1 |
|---|---:|---:|---:|---:|---:|
| SGTONetV6DualOverride | 0.7102 | 0.6233 | 0.6731 | 0.5411 | 0.5556 |
| iTransformer | 0.8175 | 0.6185 | 0.6594 | 0.4615 | 0.0000 |
| DLinear | 0.7895 | 0.5961 | 0.6466 | 0.4426 | 0.0000 |
| TimesNet | 0.7958 | 0.5893 | 0.6207 | 0.4275 | 0.0000 |
| PatchTST | 0.7559 | 0.5687 | 0.6154 | 0.4194 | 0.0000 |
| SGTONetV4Conservative | 0.7630 | 0.5605 | 0.5961 | 0.3999 | 0.0000 |

### 5. Conclusion (~0.5 pages)
- **Summary**: SGTONetV6 addresses rare-boundary collapse in short-horizon Hoister future-state prediction by separating conservative future-state classification from boundary-constrained rare triggering.
- **Limitations**: Evidence is from one private dataset, the method has lower overall accuracy than iTransformer, and the current rare-trigger calibration does not transfer cleanly to `label_shift=3`.
- **Future**: Add public or multi-site validation, improve horizon-specific calibration, and evaluate deployment-oriented false-alarm costs.

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
1. DLinear: verify official paper citation before final writing.
2. TimesNet: verify official paper citation before final writing.
3. iTransformer: verify official paper citation before final writing.
4. PatchTST: verify official paper citation before final writing.
5. Industrial fault diagnosis and predictive maintenance references: add after literature search.
6. Class imbalance, focal loss, and rare-event detection references: add after literature search.
7. Early time-series classification and lead-time fault prediction references: add after literature search.
