# CLAUDE.zh.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作提供指导。

## 概述

时序库（TSLib）是一个深度学习时间序列分析基准，涵盖五大任务：长期预测、短期预测、缺失值填充、异常检测和分类。包含 30+ 个模型，包括 TimesNet、iTransformer、PatchTST、TimeMixer 等。

## 命令

### 环境配置
```bash
conda create -n tslib python=3.11
conda activate tslib
pip install -r requirements.txt
```

### 训练与评估
```bash
# 指定参数单次运行
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model TimesNet \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7

# 运行实验脚本（推荐）
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
```

### 各任务命令
```bash
# 长期预测
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh

# 短期预测 (M4)
bash ./scripts/short_term_forecast/TimesNet_M4.sh

# 缺失值填充
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh

# 异常检测
bash ./scripts/anomaly_detection/PSM/TimesNet.sh

# 分类
bash ./scripts/classification/TimesNet.sh

# 零样本预测 (LTSMs)
bash ./scripts/long_term_forecast/ETT_script/LTSM.sh
```

### 关键参数
- `--task_name`: `long_term_forecast`, `short_term_forecast`, `imputation`, `anomaly_detection`, `classification`, `zero_shot_forecast`
- `--is_training`: `1` 训练+测试，`0` 仅测试
- `--model`: 模型名称，需与 `exp/exp_basic.py:model_dict` 中的键匹配
- `--data`: 数据集键名，来自 `data_provider/data_factory.py:data_dict`（ETTh1, ETTh2, ETTm1, ETTm2, custom, m4, PSM, MSL, SMAP, SMD, SWAT, UEA）
- `--features`: `M`（多变量→多变量），`S`（单变量→单变量），`MS`（多变量→单变量）
- `--seq_len`: 输入序列长度
- `--pred_len`: 预测长度
- `--enc_in`, `--dec_in`, `--c_out`: 输入/输出变量数量

## 架构

### 入口流程
`scripts/*.sh` → `run.py` → `Exp_*` 类（由 `task_name` 决定）→ `data_provider` + `models` → 检查点保存至 `./checkpoints/`

### 核心目录
- **`exp/`**: 任务流水线。`Exp_Basic` 注册模型；子类（`Exp_Long_Term_Forecast` 等）实现 `_get_data()`、`train()`、`test()`
- **`models/`**: 模型架构。每个文件自包含。新模型添加到此处并在 `Exp_Basic.model_dict` 中注册
- **`layers/`**: 可复用组件（注意力机制、嵌入层、分解模块），供多个模型共享
- **`data_provider/`**: 数据集加载器。`data_factory.py` 将数据集名称映射到加载类；`data_loader.py` 处理滑窗
- **`scripts/`**: 包含论文配置的 Bash 脚本，按任务/数据集/模型组织
- **`utils/`**: 评估指标（`metrics.py`）、早停（`tools.py`）、数据增强、掩码处理

### 添加新模型
1. 参照 `models/Transformer.py` 创建 `models/YourModel.py`
2. 在 `exp/exp_basic.py` 中添加导入并加入 `model_dict`
3. 在 `scripts/<task>/<dataset>/YourModel.sh` 下创建脚本

### 数据集结构
将数据集放置于 `./dataset/`。按 README 预期结构：
- ETT: `./dataset/ETT-small/ETTh1.csv`
- 异常检测: `./dataset/PSM/`、`./dataset/MSL/` 等
- 分类: `./dataset/Heartbeat/`

## 模型分类

**预测模型**: TimesNet, iTransformer, PatchTST, TimeMixer, TimeXer, DLinear, Autoformer, Informer, FEDformer, Transformer

**大型时序模型（零样本）**: Chronos, Chronos2, Moirai, TimesFM, TimeMoE, Sundial, TiRex

**异常检测**: TimesNet, KANAD (KAN-AD)

**特殊依赖**:
- Mamba: 需要 `mamba_ssm` 包（CUDA 版本特定的 wheel）
- Moirai: 需要 `pip install uni2ts --no-deps`

## 核心模型设计

### TimesNet (ICLR 2023)
**核心创新**: 时序2D变化建模
- 使用FFT发现时间序列中的主要周期
- 按周期将1D序列折叠成2D张量（如周期=24: [1,168] → [7,24]）
- 使用2D卷积（Inception块）捕获周期内变化和周期间变化
- 根据FFT幅度加权聚合多个周期的结果
- 统一框架，支持全部5大任务

### iTransformer (ICLR 2024)
**核心创新**: 倒置Transformer架构
- 传统方式: 在时间步之间做注意力
- iTransformer: 在变量（通道）之间做注意力
- 每个变量的整个时间序列成为一个token: [B,L,N] → [B,N,D]
- 更好地捕获多变量之间的依赖关系
- 避免长序列上的注意力稀释问题

### PatchTST (ICLR 2023)
**核心创新**: 分块 + 通道独立
- 将时间序列分割成固定长度的patch（类似ViT处理图像的方式）
- 每个变量独立处理，共享Transformer参数
- 减少序列长度，降低计算成本
- 关键参数: `patch_len=16`, `stride=8`（50%重叠）

### Autoformer (NeurIPS 2021)
**核心创新**: 序列分解 + 自相关机制
- 使用移动平均将序列分解为趋势项和季节项
- 自相关机制替代自注意力
- 利用周期性进行信息聚合
- O(L log L) 复杂度

### Informer (AAAI 2021 最佳论文)
**核心创新**: ProbSparse注意力 + 蒸馏
- ProbSparse注意力: 通过KL散度选择重要的Query，O(L log L)复杂度
- 自注意力蒸馏: 通过卷积逐层减半序列长度
- 生成式解码器: 一次性输出整个预测序列（非自回归）

### DLinear (AAAI 2023)
**核心创新**: 极简线性模型
- 质疑Transformer在时序预测中的有效性
- 简单的分解 + 两个线性层（趋势 + 季节）
- 参数极少，训练速度快
- `individual=True`: 每个变量单独使用线性层
