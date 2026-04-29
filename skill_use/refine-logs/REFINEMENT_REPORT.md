# Refinement Report

**Date**: 2026-04-16
**Overall Verdict**: READY
**Overall Score**: 8.9 / 10

## What Changed Relative to the Previous Proposal
1. The task is no longer framed as same-time classification with noisy windows.
2. The paper story is now centered on lead-time classification with temporal label shift.
3. The dominant bottleneck changed from generic window ambiguity to label-feature misalignment at transition boundaries.
4. The method changed from purity-aware within-window supervision to shift-aware boundary supervision.

## Final Locked Decisions
- Main task: lead-time Hoister 5-class classification with `x_{t-L+1:t} -> y_{t+Δ}`
- Main horizon: `Δ=1`
- Supporting horizons: `Δ=0`, `Δ=5`
- Main backbones: `TimesNet`, `iTransformer`, `DLinear`
- Main method type: training-only supervision change
- Main evaluation lens: fault-sensitive classification metrics, not forecasting metrics

## Why Simpler Is Better Here
The user does not need a forecasting system. The user needs a classification system robust to temporal label shift. A training-time supervision change directly addresses that requirement and is easier to defend than a hybrid classification-forecasting model.

## Remaining Risks
- Runtime CUDA visibility remains inconsistent with the user's claim.
- Fixed lead-time `Δ` may not map perfectly to real actuation or labeling delay.
- Public literature on early classification exists, but direct industrial lead-time fault classification is still a narrower niche; positioning must stay precise.

## Recommendation
Proceed to experiment planning with `Δ=1` as the main claim, and treat larger `Δ` only as robustness / scope probing.
