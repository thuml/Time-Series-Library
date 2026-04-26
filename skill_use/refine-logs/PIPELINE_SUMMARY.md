# Pipeline Summary

**Problem**: Hoister 私有工业多变量时序中的时间错拍故障分类
**Final Method Thesis**: Use shift-aware boundary supervision so current windows can classify future fault/state labels under fixed lead time without turning the task into forecasting.
**Final Verdict**: READY
**Date**: 2026-04-16

## Final Deliverables
- Proposal: `skill_use/refine-logs/FINAL_PROPOSAL.md`
- Review summary: `skill_use/refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `skill_use/refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `skill_use/refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- Dominant contribution:
  - shift-aware boundary supervision for fixed lead-time classification
- Optional supporting contribution:
  - a compact `Δ=0 / Δ=1 / Δ=5` lead-time evaluation protocol
- Explicitly rejected complexity:
  - forecasting / RUL framing, new backbones, hybrid decoder systems, benchmark sprawl

## Must-Prove Claims
- C1: `x_{t-L+1:t} -> y_{t+Δ}` is a valid and learnable classification task on Hoister data.
- C2: Boundary-aware supervision beats plain shifted-label training on the private 5-class task.
- C3: The gain persists across existing backbones without extra inference-time cost.

## First Runs to Launch
1. `DLinear` sanity run with `Δ=0` and `Δ=1` to verify loader and shifted target indexing.
2. `TimesNet` shifted hard-label baseline at `Δ=1`.
3. `TimesNet` proposed shift-aware boundary supervision at `Δ=1`.

## Main Risks
- GPU visibility mismatch:
  - The user reports GPU availability, but the current shell still does not expose CUDA.
- Horizon risk:
  - `Δ=1` may be learnable while larger lead times fail.
- Label semantics:
  - Exact business meaning of classes `1/3/5/7/9` is still not fully documented.

## Next Action
- Proceed to `/run-experiment` after implementing `label_shift` and boundary-window supervision in the data/training path.
