# Research Proposal: Shift-Aware Boundary Supervision for Time-Shifted Hoister Fault Classification

## Problem Anchor
- Bottom-line problem: 在 Hoister 私有工业多变量时序上做“时间错拍故障分类”，即用当前时刻或当前窗口输入，预测下一时刻或未来 `Δ` 步的运行状态类别。
- Must-solve bottleneck: 直接把标签前移成 `x_t -> y_{t+Δ}` 会把大量靠近状态切换点的窗口变成“未来标签正确、当前特征未完全显现”的高噪声样本，尤其会伤害少数故障类。
- Non-goals: 不做 forecasting / RUL；不发明新 backbone；不做大规模公开 benchmark 铺开；不引入与主问题无关的 foundation model 组件。
- Constraints: 复用现有 TSLib 分类代码；改动尽量集中在 CSV loader、loss 和训练脚本；私有数据 5 类严重不均衡；当前 shell 中 CUDA 不可见，执行环境需后续核验。
- Success condition: 在 `x_{t-L+1:t} -> y_{t+Δ}` 任务上，较 plain shifted-label baseline 显著提升 `macro-F1`、`balanced_accuracy`、`fault_macro_f1`、`class9_recall`，并且不增加推理时复杂度。

## Technical Gap
现有项目天然支持分类，但默认监督定义是“窗口输入 -> 窗口内 hard label”。你的新任务需要“窗口输入 -> 未来类别”。代码层面这一点并不复杂：在 loader 中加入 `label_shift` 即可生成 `y_{t+Δ}`。真正的问题在于：

1. 未来标签相对当前窗口存在系统性时间错拍。
2. 切换点附近的样本会天然变难，plain CE 会把这些边界样本当成和稳态样本一样的干净监督。
3. Hoister 5 类数据极不均衡，未来态又会进一步放大少数类和边界态学习难度。
4. 文献中很多 fault prognosis 工作实际上在做 forecasting / RUL，而不是固定 lead-time multi-class classification。

## Method Thesis
- One-sentence thesis: 用 shift-aware boundary supervision 显式区分稳态窗口与转移窗口，在不改变 backbone 的前提下，让 `x_{t-L+1:t} -> y_{t+Δ}` 的 lead-time fault classification 更稳定、更适合不平衡工业数据。
- Why this is the smallest adequate intervention: 只改监督构造和训练权重，不改模型主体。
- Why this route is timely: 它抓住的是工业部署里的 label-feature misalignment，而不是泛化地做更大的时序模型。

## Contribution Focus
- Dominant contribution: 一个面向时间错拍分类的 shift-aware boundary supervision 训练机制。
- Optional supporting contribution: 一个把 `Δ=0`、`Δ=1`、`Δ=5` 放在同一评价框架下的 lead-time classification 协议。
- Explicit non-contributions: forecasting, sequence generation, RUL, large-scale multimodal fusion。

## Proposed Method
### Complexity Budget
- Frozen / reused backbone: `TimesNet`, `iTransformer`, `DLinear`。
- New trainable components: 无新增主模块；只允许一个轻量 boundary-aware weighting / soft-target path。
- Tempting additions intentionally not used: decoder forecasting head, transition detector side branch, graph module, retrieval memory。

### System Overview
1. 在 `CSV_CLS` loader 中加入 `label_shift=Δ`，把每个窗口的目标定义为未来标签 `y_{t+Δ}`。
2. 同时保留当前端点标签 `y_t` 和窗口内部标签序列，标识该窗口是否为“transition window”。
3. 对稳态窗口使用未来 hard label 训练；对 transition window 使用 current-to-future 的 boundary-aware soft target。
4. 结合类别权重训练现有 backbone。
5. 推理仍然是标准分类：输入当前窗口，输出未来类别。

### Core Mechanism
- Shifted target: `target_shift = y_{t+Δ}`。
- Current anchor: `anchor_now = y_t`。
- Boundary indicator: 当 `y_t != y_{t+Δ}` 时，窗口视为 transition window。
- Soft target for transition windows:
  `t_w = beta * one_hot(y_t) + (1 - beta) * one_hot(y_{t+Δ})`
  其中 `beta` 可固定，也可由窗口内未来态占比估计。
- Window importance:
  transition windows 稀缺且关键，训练时单独加权；稳态窗口用标准 shifted label。
- Final loss:
  class-balanced CE / focal CE on stable windows + boundary soft CE on transition windows。

### Why This Is the Main Novelty
- Plain label shift 只是任务重定义，不足以形成强 paper。
- 真正的新意在于：把“lead-time classification 的边界样本噪声”作为主瓶颈，并用一个最小监督机制处理它。
- 这与上一版“窗口内标签分布 soft supervision”不同：上一版面向同时刻滑窗标签不纯；这一版面向未来标签与当前特征的时序错拍。

### Literature Grounding
- fault prognosis / early warning 文献很多，但大量工作走向 forecasting / RUL，而非固定 `Δ` 的多类分类。
- early / anticipatory time-series classification 提供任务动机，但工业 fault diagnosis 场景下，针对 delayed labels + imbalance + fixed lead-time classification 的工作并不密集。
- weak supervision / noisy label 文献说明 boundary ambiguity 可以通过软标签与重加权处理，但需要落到你的 lead-time classification 场景中。

### Training Plan
- 主任务: `running_state_five_class`
- lead-time settings: `Δ=1` 为主，`Δ=0` 和 `Δ=5` 作为对照
- backbone set: `TimesNet`, `iTransformer`, `DLinear`
- feature policy: 默认去掉 `id`, `time`, 两个标签列；测试 `JianSuDuan_ChaoSu` 是否泄漏
- split policy: file-level split only

### Failure Modes and Diagnostics
- If `Δ=1` 的 plain shifted baseline 已经很强，且 boundary method 没提升，说明错拍并非核心瓶颈。
- If transition windows 占比太低，方法收益可能只体现在极少数样本上。
- If `Δ` 增大后性能整体崩塌，说明数据只支持 very-short-horizon lead-time classification，而不是更长提前量。

### Novelty and Elegance Argument
该方案不把问题偷换成 forecasting，也不把分类任务复杂化成多头系统。它只抓住一个部署真实问题：标签相对输入有固定时间错拍。然后用最小的监督改造，让现有分类 backbone 变成 lead-time classifier。
