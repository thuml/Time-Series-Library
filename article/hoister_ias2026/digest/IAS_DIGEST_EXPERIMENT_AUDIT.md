# IAS Digest Experiment Audit for SGTONet

Date: 2026-04-26

## Venue Format Constraints

Source template: `article/IAS/IAS_AM2026_Digest_Template_word.docx`.

- Anonymous digest; no author names or affiliations.
- Abstract must be at most 150 words.
- Digest body from Abstract through References must be 3 to 5 pages.
- Letter paper, single column, 12 pt font, 1.65 line spacing, 1 inch margins.
- Figures and tables should appear after first citation, not at the end.
- References must be included within the 5 page limit.

The existing `article/hoister_ias2026/en/main_sgtonet.tex` is an IEEE conference-style two-column paper entry and does not match the IAS digest template. The digest therefore needs a separate single-column source.

## Current Digest Claim

Use the SGTONet line, not the older SGPH-Net hazard/time-bucket line. The paper-facing method name is SGTONet; the current repository implementation is `models/SGTONetV6.py`.

> SGTONet improves short-horizon hoister future-state classification under severe rare-fault imbalance by decoupling conservative multiclass prediction from boundary-constrained rare-fault triggering.

## Abstract Claim-to-Evidence Matrix

| Abstract statement | Required evidence | Current source | Status |
|---|---|---|---|
| Hoister task is a short-horizon future-state classification problem. | Dataset/protocol with `label_shift=1`, `seq_len=96`, file-level split seeds. | `results/sgto_v6_dual/EXPERIMENT_SUMMARY.md`; `scripts/run_future_state_multisplit.py`; `data_provider/csv_classification_loader.py` | Present |
| Dataset has 27 files and 37,417 timestamps. | Dataset facts and class counts. | `article/hoister_ias2026/SGTONETV6_MANUSCRIPT_DRAFT.md`; root CSV count excluding `experiment_outputs` is 27. | Present |
| Strong baselines can miss rare class 9. | Class-9 F1 for DLinear, TimesNet, iTransformer, PatchTST, SGTONetV4Conservative. | `results/sgto_v6_dual/full_d1_main_comparison.csv`; `fig2_class9_prf1.pdf` | Present |
| SGTONet obtains best macro-F1 among tested short-horizon models. | Three-split mean macro-F1 table. | `results/sgto_v6_dual/full_d1_main_comparison.csv`; `fig1_main_d1_metrics.pdf` | Present |
| SGTONet recovers class 9 with F1 0.5556. | Three-split mean class9 precision/recall/F1. | `full_d1_main_comparison.csv`; `fig2_class9_prf1.pdf`; confusion matrices in `/tmp/sgto_v6_dual_multisplit_clean/results/*/cm.csv` | Present |
| Rare override, boundary constraint, fallback prior, and patch-attentive rare context matter. | Deletion ablation table. | `results/sgto_v6_dual/final_main_and_ablations.csv`; `fig3_ablation.pdf` | Present |
| The method should not be claimed as multi-horizon superior. | Horizon-3 comparison. | `results/sgto_v6_dual/horizon1_vs_horizon3_summary.csv`; `fig4_horizon_transfer.pdf` | Present |

## Required Digest Tables and Figures

| Item | Purpose | Source | Status |
|---|---|---|---|
| Table I: dataset/protocol summary | Shows real-world data and evaluation setup. | Dataset facts + split protocol. | Present; add to digest text/table. |
| Table II: main comparison | Supports macro-F1 and class-9 recovery claim. | `full_d1_main_comparison.csv`. | Present. |
| Fig. 1: SGTONet overview | Explains conservative classifier plus constrained rare trigger. | Generated method overview figure. | Present. |
| Fig. 2: rare-class comparison | Shows class-9 precision/recall/F1. | `fig2_class9_prf1.pdf`. | Present. |
| Fig. 3: ablation | Mechanism evidence. | `fig3_ablation.pdf`. | Present. |
| Optional Fig.: confusion matrix | Error-pattern evidence. | `fig5_confusion_v6_vs_itransformer.pdf`. | Present; include only if page budget allows. |
| Optional Fig.: threshold sensitivity | Calibration evidence. | `fig6_threshold_sensitivity.pdf`. | Present; include only if page budget allows. |

## Experiments Not Needed for the Current Abstract

The older `article/hoister_ias2026/EXPERIMENT_REQUIREMENTS.md` asks for SGPH-Net hazard heads, time-to-fault buckets, and warning horizons `H=5` and `H=10`. These are required only if the paper returns to the old SGPH warning/digest story. They are not required for the current SGTONet abstract because the abstract does not claim event-level warning, time-to-fault prediction, or multi-horizon superiority.

## Missing Work Before Submission

No additional training experiment is required for the current scoped SGTONet digest claim. The remaining required work is document production:

1. Generate a method overview figure for SGTONet.
2. Rewrite the abstract to at most 150 words.
3. Use a single-column IAS digest source, not IEEEtran two-column format.
4. Compile or export to PDF with a toolchain that preserves anonymous PDF metadata.
5. Visually check page count, figure placement, and that references fit within 3 to 5 pages.
