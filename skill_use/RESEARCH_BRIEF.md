# RESEARCH_BRIEF

## 1. 当前阶段定位

当前并不是“从零找 idea”，而是“基于现有代码库、现有公开数据、现有私有数据和现成 GPU 资源，收敛一个可执行的研究问题并设计首轮实验”。因此本阶段的合理目标不是追求过大的方法创新，而是先把问题锚定、基线跑通、数据风险识别清楚。

## 2. 项目资产概览

- 项目路径：`/root/zm/Time-Series-Library-meter-fault_classification_prediction`
- 运行环境：`conda` 环境 `prediction`
- 计算资源：有 GPU
- 框架基础：基于 TSLib 的统一时序任务框架
- 已支持任务：long-term forecasting, short-term forecasting, imputation, anomaly detection, classification
- 当前最相关任务：`classification`
- 关键入口：`run.py`
- 关键分类实验入口：`exp/exp_classification.py`
- 关键私有数据加载器：`data_provider/csv_classification_loader.py`
- 可复用模型丰富，包含 `TimesNet`, `iTransformer`, `PatchTST`, `TimeMixer`, `TSMixer`, `DLinear`, `Mamba`, `TimeXer` 等

## 3. 数据资产概览

### 3.1 公开数据

公开数据均位于：`/root/zm/Time-Series-Library-meter-fault_classification_prediction/dataset`

从研究方向探索角度，当前最值得优先关注的是两类：

- 分类类公开集：`Heartbeat`, `UWaveGestureLibrary`, `FaceDetection`, `JapaneseVowels`, `SpokenArabicDigits`, `Handwriting`, `SelfRegulationSCP1`, `SelfRegulationSCP2`, `EthanolConcentration`, `PEMS-SF`
- 相邻任务公开集：`PSM`, `SWaT`, `SMD`, `MSL`, `SMAP`

初步判断：
- 如果主线研究问题是“私有工业时序运行状态/故障分类”，公开分类集更适合做泛化补充验证。
- 异常检测类数据可以提供灵感，但不应在主线尚未收敛时强行混入主要实验矩阵。

### 3.2 私有数据

私有数据路径：`/root/zm/Time-Series-Library-meter-fault_classification_prediction/dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579`

已确认事实：
- 数据由 `27` 个 CSV 文件组成
- 单文件长度范围约 `900` 到 `3224`
- 单文件典型长度约 `1350`
- 每个 CSV 有 `20` 列
- 当前观察到的列名如下：
  - `id`
  - `time`
  - `SuDuMoNiLiang`
  - `FPLCSuDu`
  - `BianmaQiSuDu`
  - `CSJSuDu`
  - `LGuanLongShenDu`
  - `FPLCShenDu`
  - `BianmaQiShenDu`
  - `DianshuDianliu1`
  - `DianshuDianliu2`
  - `LiCi_Current`
  - `ZhiDongPressure`
  - `ZhuLingDWQ1`
  - `ZhaBaDWQ1`
  - `WZhuJiLiang`
  - `WFuJiLiang`
  - `JianSuDuan_ChaoSu`
  - `running_state_class`
  - `running_state_five_class`

标签候选：
- `running_state_class`：3 分类，取值为 `[1, 3, 5]`
- `running_state_five_class`：5 分类，取值为 `[1, 3, 5, 7, 9]`

`running_state_five_class` 总体标签分布：
- `1: 10959`
- `3: 5214`
- `5: 15695`
- `7: 5364`
- `9: 185`

初步判断：
- 这是一个明显的类别不平衡问题，尤其 `9` 类极少。
- 如果直接拿普通 accuracy 当主指标，研究结论大概率会失真。
- 5 分类应视为主任务候选，3 分类可作为简化设置或消融，而不是反过来。

## 4. 代码层面已知约束

从 `data_provider/csv_classification_loader.py` 可确认：

- 加载器递归读取 `root_path` 下所有 `*.csv`
- 默认标签列名是 `label`
- 私有数据并没有 `label` 列，因此运行时必须显式指定 `--label_col`
- 支持 `--drop_cols`，因此 `id` 和 `time` 可以从特征中排除
- 支持 `--window_step`，说明当前任务是基于滑动窗口构造样本
- 支持 `--train_ratio` / `--val_ratio` / `--file_split_mode`
- 当前切分是按“文件”做 train/val/test，而不是按窗口随机打散，这对避免信息泄漏是正确方向，但仍需检查文件之间是否存在相同工况段分布偏差
- 支持 `window_label_mode=last|majority`，这天然提供了一个低成本但重要的实验维度
- 支持基于类频率生成 `class_weights`，并支持 `minority_boost` 等机制

这意味着：
- 研究方向最好优先围绕“滑窗分类 + 类别不平衡 + 工况泛化 + 特征选择”来设计
- 不应一开始就设计一个完全脱离现有数据管线的新范式

## 5. 当前最合理的主问题候选

### 主问题候选

面向私有 Hoister 工业时序数据的多变量运行状态/故障分类，在严重类别不平衡、窗口标签不确定、跨文件工况差异存在的条件下，如何在不显著增加工程复杂度的前提下，提高少数类识别能力与整体泛化稳定性。

### 为什么这个问题合理

- 直接对应私有数据痛点，不是空转 benchmark
- 能复用现有分类管线与大量 baseline
- 容易组织为论文中的清晰贡献链：问题定义 -> 方法 -> 不平衡鲁棒性 -> 泛化验证
- 即使第一版方法很克制，也能通过严谨实验形成有价值结论

## 6. 当前最值得优先验证的实验轴

优先级建议如下：

1. 标签粒度
- `running_state_five_class` 主任务
- `running_state_class` 简化任务/对照组

2. Baseline 收敛
- 第一轮优先：`TimesNet`, `iTransformer`, `PatchTST`, `TimeMixer`, `DLinear`, `TSMixer`
- 原则：先选 4 到 6 个工程上最稳、代表性最强的模型，不要把仓库里所有模型都跑一遍

3. 指标体系
- 主指标：`macro-F1`
- 辅指标：`weighted-F1`, `accuracy`, `per-class recall`, `balanced accuracy`
- 对于极少类，单独汇报召回率

4. 数据与窗口设置
- 是否丢弃 `id`, `time`
- `seq_len` 的不同取值
- `window_label_mode = last` vs `majority`
- 文件级切分策略是否稳定

5. 不平衡处理
- 无重加权 baseline
- 类别权重
- minority boost / 重采样
- 未来可考虑 focal loss 或 class-balanced loss，但不宜在第一轮就铺太开

## 7. 当前阶段可形成的研究方向雏形

### 方向 A

工业私有多变量时序故障分类中的少数类鲁棒识别。

最小闭环：
- 以 5 分类为主任务
- 以现有 backbone 为特征编码器
- 重点研究类别不平衡处理、窗口标签策略、关键变量选择

### 方向 B

跨数据集的一致时序分类归纳偏置是否能迁移到工业故障识别。

最小闭环：
- 先在私有集做主验证
- 再在 1 到 2 个公开分类集做补充对照
- 目标不是追求 SOTA，而是证明方法在“工业私有集 + 公开集”上都稳定

### 方向 C

面向工业时序片段的轻量级、可部署分类方案。

最小闭环：
- 在保证性能的同时比较参数量、推理成本、窗口长度敏感性
- 更偏工程应用，如果你后续关心在线部署，这条路值得保留

当前建议：
- 优先选方向 A 作为主线
- B 作为论文补强
- C 作为应用讨论或次级分析

## 8. 现阶段最关键的风险

- 标签语义尚未正式确认，尤其 `running_state_class` 与 `running_state_five_class` 的业务含义还不清楚
- `id` 和 `time` 是否会造成伪特征泄漏，尚未验证
- 文件级切分是否与真实业务场景一致，尚未验证
- 你“想重点比较的实验”尚未明确，如果范围过大，第一轮会分散算力
- 预测任务是否真的有标签和业务需求，尚不明确；如果没有，不应强行混在主线里

## 9. 建议的下一步

- 首先使用 `skill_use/RESEARCH_DIRECTION_PROMPT.md` 中的主提示词调用技能
- 明确告诉技能：主任务先做 `running_state_five_class`，`running_state_class` 作为简化设置或消融
- 让技能先收敛 baseline 和实验计划，再决定是否扩展到预测任务

## 10. 待你补充但不阻塞当前推进的信息

- 每个标签值对应的真实业务状态含义
- 你最关心的是识别精度、少数类召回、还是上线部署成本
- 你已经计划要比较的模型或实验
- 是否存在额外私有预测标签或故障预警标签
- 是否有必须遵守的时间预算 / GPU 预算

## 11. 当前结论

是的，当前阶段完全可以生成 `RESEARCH_BRIEF.md`。

但必须把它视为“基于现有代码与数据结构的第一版预研简报”，不是最终立题文档。它已经足够支持下一步调用技能做方向收敛、实验规划和 baseline 筛选。
