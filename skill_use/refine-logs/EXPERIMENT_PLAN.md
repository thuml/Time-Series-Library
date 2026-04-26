# Experiment Plan

**Problem**: Hoister 私有工业时序中的时间错拍故障分类
**Method Thesis**: Shift-aware boundary supervision improves fixed lead-time fault classification under temporal label misalignment and class imbalance.
**Date**: 2026-04-16

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1 | `x_{t-L+1:t} -> y_{t+Δ}` is a meaningful classification task for this dataset | `Δ=1` plain shifted baseline remains learnable and transition windows are non-trivial in frequency | B1 |
| C2 | Plain label shift is not enough because transition windows create supervision mismatch | Proposed method beats shifted hard-label baselines on `macro-F1`, `balanced_accuracy`, `fault_macro_f1` | B1, B2 |
| C3 | Gain does not come from a stronger backbone or inference cost | Consistent gains across existing backbones with unchanged inference path | B3 |
| A1 | The method is not just another imbalance tweak | Proposed method beats focal / sampler baselines under the same shifted setting | B2, B4 |

## Paper Storyline
- Main paper must prove:
  - fixed lead-time classification is the right abstraction for this project
  - transition windows are the main source of difficulty after label shift
  - boundary-aware supervision improves the private 5-class task without architectural inflation
- Appendix can support:
  - `Δ=5` results
  - 3-class simplified task
  - multi-split robustness
- Experiments intentionally cut:
  - forecasting baselines
  - RUL baselines
  - large public-benchmark coverage

## Experiment Blocks

### Block 1: Task Validation and Main Anchor Result
- Claim tested: C1, C2
- Why this block exists: 先证明 lead-time classification 本身不是伪任务，然后证明 proposed supervision 有价值
- Dataset / split / task:
  - Dataset: `dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579`
  - Label: `running_state_five_class`
  - Split: file-level, fixed `split_seed`
  - Settings: `Δ=0`, `Δ=1`
- Compared systems:
  - `TimesNet` same-time (`Δ=0`) baseline
  - `TimesNet` shifted hard label (`Δ=1`)
  - `TimesNet` shifted boundary-aware supervision (`Δ=1`)
- Metrics:
  - primary: `macro-F1`, `balanced_accuracy`
  - secondary: `fault_macro_f1`, `class9_recall`, `accuracy`
- Setup details:
  - `seq_len=96`, `window_step=8`
  - `label_shift in {0,1}`
  - `drop_cols=id,time,running_state_class,running_state_five_class`
- Success criterion:
  - `Δ=1` shifted baseline is materially above trivial majority behavior
  - proposed method beats shifted hard-label baseline on the main metrics
- Failure interpretation:
  - If `Δ=1` itself is barely learnable, the paper should retreat to same-time classification or redefine the lead time
- Table / figure target:
  - Main Table 1
  - Figure: confusion matrix or class-wise recall for `Δ=1`
- Priority: MUST-RUN

### Block 2: Boundary vs Plain Shift vs Imbalance Tricks
- Claim tested: C2, A1
- Why this block exists: 隔离“边界监督”是不是核心，而不是简单重加权
- Dataset / split / task:
  - Same Hoister 5-class setting
  - `Δ=1`
- Compared systems:
  - shifted hard label + CE
  - shifted hard label + focal
  - shifted hard label + balanced sampler + focal
  - proposed shift-aware boundary supervision
  - proposed + focal / sampler overbuilt variant
- Metrics:
  - `macro-F1`, `fault_macro_f1`, `class9_f1`, `class9_recall`
- Setup details:
  - same backbone, same seed, same split
- Success criterion:
  - proposed method clearly beats the strongest shifted hard-label imbalance baseline
- Failure interpretation:
  - If focal/sampler already closes the gap, the novelty shrinks to a data reweighting recipe
- Table / figure target:
  - Main Table 2
- Priority: MUST-RUN

### Block 3: Cross-Backbone Transfer with Unchanged Inference Path
- Claim tested: C3
- Why this block exists: 防止 reviewer 说只是某个 backbone 偶然有效
- Dataset / split / task:
  - Hoister 5-class
  - `Δ=1`
- Compared systems:
  - `TimesNet`
  - `iTransformer`
  - `DLinear`
  - each with shifted hard label vs proposed supervision
- Metrics:
  - `macro-F1`, `balanced_accuracy`, train time, inference time
- Setup details:
  - align training epochs and batch sizes as fairly as possible
- Success criterion:
  - directionally consistent gains on at least 2 model families
- Failure interpretation:
  - If gains only hold on one backbone, the story becomes model-coupled
- Table / figure target:
  - Main Table 3 or merged into Table 1
- Priority: MUST-RUN

### Block 4: Horizon and Simplicity Check
- Claim tested: C1, A1
- Why this block exists: 确定 paper 该 claim 多长 lead time，以及方法有没有被过度设计
- Dataset / split / task:
  - Hoister 5-class
  - `Δ in {1,5}`
- Compared systems:
  - shifted hard label (`Δ=1`, `Δ=5`)
  - proposed method (`Δ=1`, `Δ=5`)
  - proposed without boundary soft target
  - proposed without transition upweighting
- Metrics:
  - `macro-F1`, `balanced_accuracy`, `class9_recall`
- Setup details:
  - best backbone only
- Success criterion:
  - method works best at `Δ=1`; `Δ=5` serves as scope limit or robustness probe
  - removing boundary logic hurts results at `Δ=1`
- Failure interpretation:
  - If `Δ=5` collapses, explicitly claim short-horizon lead-time classification only
  - If deletion variants match the full method, simplify the proposal further
- Table / figure target:
  - Main or appendix ablation table
- Priority: MUST-RUN

### Block 5: Robustness to Split and Seed Variation
- Claim tested: robustness support
- Why this block exists: 文件数只有 27，必须防止 split-specific 偶然结论
- Dataset / split / task:
  - Hoister 5-class, `Δ=1`
  - multiple seeds, optionally multiple split seeds
- Compared systems:
  - strongest shifted hard baseline
  - proposed method
- Metrics:
  - mean/std of `macro-F1`, `balanced_accuracy`, `fault_macro_f1`
- Setup details:
  - at least 3 random seeds
- Success criterion:
  - proposed mean improvement persists without unstable variance explosion
- Failure interpretation:
  - If variance dominates gains, tone down claims
- Table / figure target:
  - Appendix robustness table
- Priority: NICE-TO-HAVE after core positive signal

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | sanity for shifted task | `DLinear` with `Δ=0` and `Δ=1` | confirm loader and metrics work under label shift | Low | wrong target indexing |
| M1 | anchor baseline | `TimesNet` shifted hard baseline, `Δ=1` | if learnability is too low, rethink task setup | Low-Med | task may be too hard |
| M2 | first proof of method | `TimesNet` shifted hard vs proposed | if no gain, rethink boundary supervision | Medium | transition logic too weak |
| M3 | isolate against imbalance tricks | focal / sampler / proposed | if proposed not better, novelty weakens | Medium | method reduces to reweighting |
| M4 | cross-backbone validation | `iTransformer`, `DLinear` | require consistent directional gain | Medium | gains model-specific |
| M5 | horizon + robustness | `Δ=5`, multi-seed | decide final scope claim | Medium-High | compute + variance |

## Compute and Data Budget
- Total estimated GPU-hours:
  - M0-M2: ~6-10 GPU-hours on a single accessible GPU
  - Full must-run set with 3 seeds: ~18-30 GPU-hours depending on backbone mix
- Data preparation needs:
  - add `label_shift`
  - add future-target indexing guardrails
  - expose `current_label`, `future_label`, `is_transition_window`
- Human evaluation needs:
  - none
- Biggest bottleneck:
  - real CUDA availability on the intended training machine

## Risks and Mitigations
- Risk: current shell cannot see CUDA
  - Mitigation: verify on the actual training host before long runs
- Risk: `Δ=1` works but `Δ=5` fails badly
  - Mitigation: lock the paper scope to short-horizon lead-time classification
- Risk: derived feature leakage via `JianSuDuan_ChaoSu`
  - Mitigation: mandatory with/without comparison in the first ablation wave
- Risk: boundary windows are not the real source of error
  - Mitigation: compare against plain shift + focal/sampler before overcommitting to the method

## Final Checklist
- [ ] Main paper tables are covered
- [ ] Novelty is isolated
- [ ] Simplicity is defended
- [x] Forecasting / RUL claims are explicitly excluded
- [ ] Nice-to-have runs are separated from must-run runs
