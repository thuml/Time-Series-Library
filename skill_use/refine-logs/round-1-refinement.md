# Refined Proposal Snapshot

## Locked Problem Anchor
在 Hoister 私有工业时序上做时间错拍故障分类：输入当前窗口，预测未来 `Δ` 步的状态类别，而不是做 forecasting 或 RUL。

## Final Method Thesis
Shift-Aware Boundary Supervision: 在 label-shift 分类中显式识别 transition windows，并对这些窗口采用 current-to-future 的边界软监督与重加权训练。

## Dominant Contribution
一个面向 `x_{t-L+1:t} -> y_{t+Δ}` lead-time classification 的训练期监督机制，能在现有 backbone 上低成本落地。

## Complexity Intentionally Rejected
- forecasting decoder
- multi-task classification + forecasting
- explicit transition head
- graph / memory / MoE add-ons
- public-benchmark-heavy story

## Must-Run Claims
1. Lead-time classification (`Δ>0`) 在 Hoister 数据上是合理且可学的，不是伪任务。
2. Proposed boundary-aware supervision beats plain shifted-label baselines on the private 5-class task.
3. Gains persist across multiple backbones without extra inference-time cost.

## Remaining Risks
- actual CUDA availability is still unverified in the current shell
- `Δ=1` 可能最合理，但更大 `Δ` 不一定可学
- label semantics for each class remain partially undocumented
