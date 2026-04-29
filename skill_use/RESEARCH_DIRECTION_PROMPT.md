# 研究方向探索提示词

以下提示词面向 ARIS / Codex 风格技能，当前阶段优先建议使用 `/research-refine-pipeline`，而不是纯 `/idea-discovery`。原因很简单：你已经有项目代码、公开数据、私有数据、可复用模型和 GPU，不是从零找题，而是应该围绕已有资产收敛成一个可执行、可验证、可发表的方案。

## 推荐主提示词

```text
/research-refine-pipeline "请基于项目 /root/zm/Time-Series-Library-meter-fault_classification_prediction 进行研究方向探索，并收敛为一个问题锚点明确、贡献集中的可执行研究方案，再输出实验计划。请优先结合现有代码、已有模型、可直接运行的数据和 GPU 条件，而不是泛泛 brainstorm。

项目与环境：
- 项目路径：/root/zm/Time-Series-Library-meter-fault_classification_prediction
- Python 环境：conda 环境 prediction
- 计算资源：有 GPU，可运行中等规模实验
- 目标文件夹：/root/zm/Time-Series-Library-meter-fault_classification_prediction/skill_use

数据情况：
- 所有数据位于：/root/zm/Time-Series-Library-meter-fault_classification_prediction/dataset
- 私有数据位于：/root/zm/Time-Series-Library-meter-fault_classification_prediction/dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579
- 该私有数据集由 27 个 CSV 文件组成，每个文件代表一个运行片段
- 单文件长度约 900 到 3224，典型长度约 1350
- CSV 共有 20 列：id、time、17 个左右传感器/过程变量、2 个标签列
- 当前观测到的列包括：id, time, SuDuMoNiLiang, FPLCSuDu, BianmaQiSuDu, CSJSuDu, LGuanLongShenDu, FPLCShenDu, BianmaQiShenDu, DianshuDianliu1, DianshuDianliu2, LiCi_Current, ZhiDongPressure, ZhuLingDWQ1, ZhaBaDWQ1, WZhuJiLiang, WFuJiLiang, JianSuDuan_ChaoSu, running_state_class, running_state_five_class
- 其中 running_state_class 是 3 类标签，取值为 [1, 3, 5]
- running_state_five_class 是 5 类标签，取值为 [1, 3, 5, 7, 9]
- 5 类标签分布严重不均衡，大致为：1:10959, 3:5214, 5:15695, 7:5364, 9:185
- 因此请把类别不平衡、少数类识别、代价敏感学习、分层评测纳入研究设计

代码与现有能力：
- 该项目基于 TSLib，已包含大量时序模型，如 TimesNet, iTransformer, PatchTST, TimeMixer, TSMixer, DLinear, Mamba, TimeXer 等
- 项目已有分类任务入口：run.py --task_name classification
- 项目已有多文件 CSV 分类数据加载器：data_provider/csv_classification_loader.py
- 该加载器支持 label_col, drop_cols, window_step, train_ratio, val_ratio, file_split_mode, window_label_mode 等参数
- 默认 label 列名是 label，但私有数据没有 label 列，真实任务中应显式指定 running_state_five_class 或 running_state_class
- id 和 time 很可能不能直接作为数值特征使用，请检查是否应该放到 drop_cols
- 当前任务更适合先聚焦“私有数据上的多变量时间序列故障/运行状态分类”，预测任务可以作为次要扩展，不要一开始把问题做散

公开数据与对比思路：
- dataset 目录下已有多个公开数据集，可用于补充验证或方法泛化分析
- 分类类公开数据可优先考虑：Heartbeat, UWaveGestureLibrary, FaceDetection, JapaneseVowels, SpokenArabicDigits, Handwriting, SelfRegulationSCP1/2, EthanolConcentration, PEMS-SF
- 异常检测类公开数据如 PSM, SWaT, SMD, MSL, SMAP 可作为相邻任务参考，但除非研究问题明确依赖异常检测，否则不要强行混入主线
- 请优先从当前仓库中可直接运行的模型里挑选 baseline，不要凭空引入一批仓外模型增加工程成本

我希望你完成以下工作：
1. 先阅读项目代码、README、run.py、data_provider/csv_classification_loader.py、exp/exp_classification.py、models/ 下与分类最相关的模型实现，以及 dataset 目录结构
2. 基于现有资产提出 3 个以内真正值得做的研究方向，每个方向都必须满足：问题明确、改动可控、实验可落地、与私有数据痛点强相关、能与现有 baseline 明确比较
3. 明确推荐一个主方向，不要给模糊的平行备选清单
4. 对主方向给出：问题锚点、核心假设、方法最小闭环、为什么比直接堆模型更合理、最关键风险
5. 输出 claim-driven 实验计划，包括：
   - 主任务与副任务
   - 数据使用方案：私有 5 类主任务，必要时用 3 类任务作简化或消融
   - baseline 列表，优先从 TimesNet、iTransformer、PatchTST、TimeMixer、DLinear、TSMixer 中筛选最合适者
   - 评价指标：macro-F1、weighted-F1、per-class recall、accuracy；若涉及不平衡学习，请加入 balanced accuracy 或 G-mean
   - 必做消融：标签粒度 3 类 vs 5 类、是否去掉 id/time、不同 seq_len、不同 window_label_mode、是否类别重加权/重采样
   - 数据划分风险：必须检查 file-level split 是否合理，避免片段泄漏
   - 首批最值得启动的 3 组实验
6. 明确指出当前还缺哪些信息会影响方案质量，例如：真实业务目标、可接受延迟、是否需要在线检测、你已经想对比的实验列表、是否有预测任务标签
7. 如果信息不足，不要停在“请补充信息”；请先基于现有信息给出一个可执行的一版方案，并把待确认项列成 TODO

输出要求：
- 在 /root/zm/Time-Series-Library-meter-fault_classification_prediction/skill_use 下产出结构化结果
- 至少包含一个简明研究简报和一个实验计划草案
- 所有建议必须尽量复用现有项目能力，避免不必要的大改代码
- 以可发表但不过度冒进为原则，避免空泛的新颖性包装"
```

## 可选备选提示词

如果你只是想先大范围发散，再缩回来，可以用这个版本：

```text
/idea-discovery "请基于 /root/zm/Time-Series-Library-meter-fault_classification_prediction 的现有代码、公开数据和私有数据，围绕 Hoister 私有数据集的多变量时间序列故障/运行状态分类任务，提出 3-5 个值得做但实验成本可控的研究方向。私有数据位于 /root/zm/Time-Series-Library-meter-fault_classification_prediction/dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579 。该数据包含 27 个 CSV 文件，主要标签候选为 running_state_five_class（5 类，类别极不均衡）和 running_state_class（3 类）。请结合项目已有模型和分类数据加载器，优先设计能直接复用现有代码的方向，并指出每个方向最适合比较的 baseline、实验难点和最小验证路径。不要只给论文式概念，请给能落地的方案。"
```

## 使用建议

- 第一轮先用主提示词。
- 如果技能先问你 label 列用哪个，优先回答：主任务先用 `running_state_five_class`，`running_state_class` 作为简化设置或消融。
- 如果技能先问你是不是要同时做预测和分类，建议回答：先把分类主线做扎实，预测只在有明确标签和业务价值时再单独立题。
- 如果技能先问你有哪些想比较的实验，优先让它从仓库已有模型里收敛到 4 到 6 个 baseline，不要一开始铺太大。
