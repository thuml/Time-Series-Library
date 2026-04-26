# Final Proposal: Shift-Aware Boundary Supervision for Time-Shifted Hoister Fault Classification

**Date**: 2026-04-16
**Status**: READY

## Problem Anchor
- Bottom-line problem:
  在 Hoister 私有工业多变量时序上做“时间错拍故障分类”：用当前时刻或当前窗口输入，预测未来 `Δ` 步的状态类别。
- Must-solve bottleneck:
  把标签简单前移成 `x_{t-L+1:t} -> y_{t+Δ}` 会在状态切换附近引入系统性监督错配。当前特征可能仍像旧状态，但目标已经被定义成未来状态。
- Non-goals:
  不做 forecasting，不做 RUL，不发明新 backbone，不把故事扩成通用 predictive maintenance 平台。
- Constraints:
  复用现有 TSLib 分类管线；重点改 `csv_classification_loader.py`、训练监督与实验脚本；私有 5 类数据不平衡；当前 shell 里 CUDA 未可见。
- Success condition:
  在 `Δ>0` 的 lead-time classification 上，相比 plain shifted-label baseline，显著提升 `macro-F1`、`balanced_accuracy`、`fault_macro_f1`、`class9_recall`，且不增加推理时复杂度。

## Technical Gap
你的任务不是普通故障分类，也不是 forecasting。它是一个夹在两者之间、但更贴近部署的设定：标签相对输入错拍，系统希望用当前传感器信息提前一个小步长判断下一状态。

当前代码的默认监督逻辑是：
- 取窗口 `x_{t-L+1:t}`
- 从窗口内部标签生成一个 hard label

而你现在需要的是：
- 取窗口 `x_{t-L+1:t}`
- 预测 `y_{t+Δ}`

如果只做 plain shift，会出现两个问题：
1. 边界窗口被硬性赋成未来态，但输入仍以当前态特征为主；
2. 在 5 类严重不平衡下，这些 transition windows 更容易放大少数类学习不稳定性。

私有数据支持这个问题设定，不是空想。基于 `running_state_five_class` 的快速统计显示：
- 共 27 个文件
- 一步标签转移率约 `0.1200`
- 每个文件平均约 `166` 次标签切换，median `147`
- 说明状态切换并不少，lead-time classification 不是只覆盖极少数边角样本

## Method Thesis
**One-sentence thesis**: Use shift-aware boundary supervision for `x_{t-L+1:t} -> y_{t+Δ}` classification, so transition windows are trained with ambiguity-aware targets instead of treated as clean future-state labels.

## Contribution Focus
- Dominant contribution:
  A shift-aware boundary supervision scheme for fixed lead-time industrial multi-class classification.
- Optional supporting contribution:
  A compact evaluation protocol comparing `Δ=0`, `Δ=1`, and `Δ=5` as same-task variants.
- Explicit non-contributions:
  forecasting, RUL estimation, generative sequence modeling, new backbone design.

## Proposed Method
### Complexity Budget
- Frozen / reused backbone:
  `TimesNet`, `iTransformer`, `DLinear`.
- New trainable components:
  None in the main version.
- Tempting additions intentionally not used:
  forecasting head, transition detector branch, graph modules, memory bank, multi-task auxiliary decoder.

### System Overview
1. Extend the CSV classification loader with `label_shift=Δ`.
2. For each window ending at time `t`, define the prediction target as `y_{t+Δ}`.
3. Also keep `y_t` and the within-window label sequence for supervision diagnostics.
4. Mark a sample as a transition window if `y_t != y_{t+Δ}`.
5. Use stable-window hard supervision and transition-window boundary-aware soft supervision.
6. Train existing backbones exactly as classifiers.
7. At inference, feed a current window and output the predicted future class.

### Core Mechanism
For a window `w = x_{t-L+1:t}`:
- Future target: `y_f = y_{t+Δ}`
- Current anchor: `y_c = y_t`
- Transition indicator: `b_w = 1[y_c != y_f]`

Supervision:
- Stable window (`b_w = 0`):
  use standard class-balanced CE on `y_f`
- Transition window (`b_w = 1`):
  use a soft boundary target
  `t_w = beta * one_hot(y_c) + (1 - beta) * one_hot(y_f)`

Interpretation:
- `beta` captures that the current window still contains old-state evidence
- the model should learn to anticipate the next state without being punished as if the transition were already complete everywhere in the window

Boundary weighting:
- transition windows receive an additional importance factor `λ_b`
- rare classes still use inverse-frequency class weights or focal-style scaling

Final loss:
`L = L_stable + λ_b * L_transition`
with class balancing applied in both terms.

### Why This Is the Smallest Adequate Intervention
- It solves the exact mismatch introduced by label shift.
- It preserves the classification backbone and inference path.
- It is easy to compare against plain shift baselines.
- It avoids claiming novelty from architecture size or extra parameters.

### Training Recipe
- Main task:
  `running_state_five_class`
- Task variants:
  - `Δ=0`: standard same-time classification baseline
  - `Δ=1`: main setting
  - `Δ=5`: robustness / appendix setting
- Features:
  start with dropping `id`, `time`, `running_state_class`, `running_state_five_class`
- Leakage check:
  test with and without `JianSuDuan_ChaoSu`
- Backbone set:
  `TimesNet`, `iTransformer`, `DLinear`
- Baseline supervision set:
  - same-time hard label (`Δ=0`)
  - plain shifted hard label (`Δ=1`)
  - shifted hard label + focal / sampler
  - proposed shift-aware boundary supervision

### Literature Positioning
This proposal is motivated by, but distinct from, three neighboring literatures:
- fault prognosis / early warning: often forecasting-like or RUL-oriented
- noisy-label / weakly supervised TSC: useful supervision tools, but not necessarily lead-time industrial classification
- imbalance-aware industrial fault diagnosis: relevant metrics and robustness concerns, but not specifically label-shifted classification

The paper story is therefore narrower and cleaner: **fixed lead-time classification under temporal misalignment**.

### Failure Modes and Diagnostics
- If `Δ=1` plain shift already works well and transition-aware supervision adds little, then time misalignment may not be the dominant bottleneck.
- If `Δ=5` collapses while `Δ=1` works, the paper should explicitly claim short-horizon lead-time classification only.
- If class `9` remains unstable, the paper must avoid overclaiming rare-class gains.
- If split variance is large, robustness reporting becomes part of the contribution.

### Novelty and Elegance Argument
The paper does not claim “we predict the future” in a generic sense. It claims something more specific and more defensible: in industrial classification settings where labels are intentionally shifted forward by a small lead time, the hardest samples are transition windows, and treating them as clean future-state labels is suboptimal. A shift-aware boundary supervision scheme is enough to fix that mismatch.

## Planning Gate
- Final method thesis: locked.
- Dominant contribution: locked.
- Complexity intentionally rejected: explicit.
- Reviewer concerns that still matter: fixed lead-time validity, transition frequency, GPU visibility, label semantics.
- Frontier primitive: absent by design.
