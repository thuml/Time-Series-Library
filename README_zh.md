# 时间序列库（TSLib）
TSLib 是一个面向深度学习研究者的开源库，特别适用于深度时间序列分析。

> **English README**：[README.md](./README.md)

我们提供了一个整洁的代码库，用于评测先进的深度时间序列模型或开发自定义模型，覆盖 **长短期预测、插补、异常检测和分类** 等五大主流任务。

:triangular_flag_on_post:**最新动态**（2025.12）非常感谢 [ailuntz](https://github.com/thuml/Time-Series-Library/pull/805) 的杰出贡献，提供了更新的依赖要求和 Docker 部署，以及完善的文档。这对本项目和初学者都很有意义。

:triangular_flag_on_post:**最新动态**（2025.11）鉴于大型时间序列模型（LTSM）的快速发展，我们在 TSLib 中新增了[[零样本预测]](https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_zero_shot_forecasting.py)功能，可参考 [此脚本](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/LTSM.sh) 评测 LTSM。

:triangular_flag_on_post:**最新动态**（2025.10）针对近期研究者在标准基准上追求微小提升而产生的困惑，我们提出了[[精度定律]](https://arxiv.org/abs/2510.02729)，以刻画深度时间序列预测任务的目标，并可据此识别已饱和的数据集。

:triangular_flag_on_post:**最新动态**（2024.10）我们已纳入 [[TimeXer]](https://arxiv.org/abs/2402.19072)，其定义了一个实用的预测范式：带外生变量的预测。考虑到实用性与计算效率，我们认为 TimeXer 所定义的新范式将成为未来研究的“正确”任务。

:triangular_flag_on_post:**最新动态**（2024.10）实验室已开源 [[OpenLTM]](https://github.com/thuml/OpenLTM)，提供了有别于 TSLib 的预训练 - 微调范式。如果您对大型时间序列模型感兴趣，该仓库值得参考。

:triangular_flag_on_post:**最新动态**（2024.07）我们撰写了关于[[深度时间序列模型]](https://arxiv.org/abs/2407.13278)的综述，并基于 TSLib 构建了严谨的基准。论文总结了当前时间序列模型的设计原则，并通过深入实验验证，期望对未来研究有所帮助。

:triangular_flag_on_post:**最新动态**（2024.04）感谢 [frecklebars](https://github.com/thuml/Time-Series-Library/pull/378) 的贡献，著名的序列模型 [Mamba](https://arxiv.org/abs/2312.00752) 已加入本库。参见[该文件](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py)，需要先用 pip 安装 `mamba_ssm`。

:triangular_flag_on_post:**最新动态**（2024.03）鉴于各论文使用的回溯窗口长度不一致，我们将排行榜中的长期预测拆分为 Look-Back-96 与 Look-Back-Searching 两类。建议阅读 [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2)，其实验同时包含两种窗口设置，更具科学性。

:triangular_flag_on_post:**最新动态**（2023.10）我们添加了 [iTransformer](https://arxiv.org/abs/2310.06625) 的实现，这是长期预测领域的最新 SOTA。官方代码与完整脚本参见 [此处](https://github.com/thuml/iTransformer)。

:triangular_flag_on_post:**最新动态**（2023.09）我们为 [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) 及本库添加了详细[教程](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb)，对时间序列初学者十分友好。

:triangular_flag_on_post:**最新动态**（2023.02）我们发布了 TSlib，作为一个面向时间序列模型的综合基准与代码库，扩展自此前的 [Autoformer](https://github.com/thuml/Autoformer) 仓库。

## 时间序列分析排行榜

截至 2024 年 3 月，各任务排行榜前三名如下：

| 模型<br>排名 | 长期预测<br>Look-Back-96 | 长期预测<br/>Look-Back-Searching | 短期预测 | 插补 | 分类 | 异常检测 |
| ------------ | ------------------------ | -------------------------------- | -------- | ---- | ---- | -------- |
| 🥇 第一名 | [TimeXer](https://arxiv.org/abs/2402.19072) | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2) | [TimesNet](https://arxiv.org/abs/2210.02186) | [TimesNet](https://arxiv.org/abs/2210.02186) | [TimesNet](https://arxiv.org/abs/2210.02186) | [TimesNet](https://arxiv.org/abs/2210.02186) |
| 🥈 第二名 | [iTransformer](https://arxiv.org/abs/2310.06625) | [PatchTST](https://github.com/yuqinie98/PatchTST) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 🥉 第三名 | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2) | [DLinear](https://arxiv.org/pdf/2205.13504.pdf) | [FEDformer](https://github.com/MAZiqing/FEDformer) | [Autoformer](https://github.com/thuml/Autoformer) | [Informer](https://github.com/zhouhaoyi/Informer2020) | [Autoformer](https://github.com/thuml/Autoformer) |

**说明：排行榜会持续更新。** 如果您提出了先进的模型，可通过发送论文或代码链接、或提交 PR 与我们联系，我们会尽快将其加入仓库并更新排行榜。

**排行榜中的对比模型**（☑ 表示代码已收录）。
  - [x] **TimeXer** - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[NeurIPS 2024]](https://arxiv.org/abs/2402.19072) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py)
  - [x] **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py)
  - [x] **TSMixer** - TSMixer: An All-MLP Architecture for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/pdf/2303.06053.pdf) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py)
  - [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py)
  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)
  - [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py)
  - [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py)
  - [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py)
  - [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py)
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py)
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py)
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py)
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py)
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py)
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py)

更多详情可参考我们关于 [[TimesNet]](https://arxiv.org/abs/2210.02186) 的最新论文，实时在线版本即将发布。

**新增基线模型**（综合评测后将加入排行榜）。
  - [x] **MambaSL** - MambaSL: Exploring Single-Layer Mamba for Time Series Classification [[ICLR 2026]](https://openreview.net/forum?id=YDl4vqQqGP) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MambaSingleLayer.py)
  - [x] **TimeFilter** - TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting [[ICML 2025]](https://arxiv.org/abs/2501.13041) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeFilter.py)
  - [x] **KAN-AD** - KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks [[ICML 2025]](https://arxiv.org/abs/2411.00278) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/KANAD.py)
  - [x] **MultiPatchFormer** - A multiscale model for multivariate time series forecasting [[Scientific Reports 2025]](https://www.nature.com/articles/s41598-024-82417-4) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/MultiPatchFormer.py)
  - [x] **WPMixer** - WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting [[AAAI 2025]](https://arxiv.org/abs/2412.17176) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/WPMixer.py)
  - [x] **MSGNet** - MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting [[AAAI 2024]](https://dl.acm.org/doi/10.1609/aaai.v38i10.28991) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/MSGNet.py)
  - [x] **PAttn** - Are Language Models Actually Useful for Time Series Forecasting? [[NeurIPS 2024]](https://arxiv.org/pdf/2406.16964) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/PAttn.py)
  - [x] **Mamba** - Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[arXiv 2023]](https://arxiv.org/abs/2312.00752) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py)
  - [x] **SegRNN** - SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [[arXiv 2023]](https://arxiv.org/abs/2308.11200.pdf) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/SegRNN.py)
  - [x] **Koopa** - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [[NeurIPS 2023]](https://arxiv.org/pdf/2305.18803.pdf) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Koopa.py)
  - [x] **FreTS** - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [[NeurIPS 2023]](https://arxiv.org/pdf/2311.06184.pdf) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/FreTS.py)
  - [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py)
  - [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py)
  - [x] **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py)
  - [x] **SCINet** - SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction [[NeurIPS 2022]](https://openreview.net/pdf?id=AyajSjTAzmg) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/SCINet.py)
  - [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py)
  - [x] **TFT** - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting [[arXiv 2019]](https://arxiv.org/abs/1912.09363) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TemporalFusionTransformer.py)

**新增大型时间序列模型**。本库同样支持以下 LTSM 的零样本评测：

- [x] **Chronos2** - Chronos-2: From Univariate to Universal Forecasting [[arXiv 2025]](https://arxiv.org/abs/2510.15821) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Chronos2.py)
- [x] **TiRex** - TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning [[NeurIPS 2025]](https://arxiv.org/pdf/2505.23719) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiRex.py)
- [x] **Sundial** - Sundial: A Family of Highly Capable Time Series Foundation Models [[ICML 2025]](https://arxiv.org/pdf/2502.00816) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Sundial.py)
- [x] **Time-MoE** - Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts [[ICLR 2025]](https://arxiv.org/pdf/2409.16040) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMoE.py)
- [x] **Toto** - Toto: Time Series Optimized Transformer for Observability [[arXiv 2024]](https://arxiv.org/pdf/2407.07874)
- [x] **Chronos** - Chronos: Learning the Language of Time Series [[TMLR 2024]](https://arxiv.org/pdf/2403.07815) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/Chronos.py)
- [x] **Moirai** - Unified Training of Universal Time Series Forecasting Transformers [[ICML 2024]](https://arxiv.org/pdf/2402.02592)
- [x] **TimesFM** - TimesFM: A decoder-only foundation model for time-series forecasting [[ICML 2024]](https://arxiv.org/abs/2310.10688) [[代码]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesFM.py)

## 快速开始

### 准备数据
可从 [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)、[[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) 或 [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library) 下载预处理数据，并置于 `./dataset` 目录。

### 安装
1. 克隆本仓库
   ```bash
   git clone https://github.com/thuml/Time-Series-Library.git
   cd Time-Series-Library
   ```

2. 创建新的 Conda 环境
   ```bash
   conda create -n tslib python=3.11
   conda activate tslib
   ```

3. 安装核心依赖
   > ⚠️ **CUDA 兼容性提示**
   > torch 预编译包与 **CUDA 版本强相关**。（查看 https://pytorch.org/get-started/previous-versions/ ）
   > 请确保torch安装与本地 CUDA 版本匹配的包（如 `cu118` 或 `cu121`）。
   > 推荐torch==2.5.1

   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   
   pip install -r requirements.txt
   ```

4. 安装 Mamba 模型依赖（models/Mamba.py 需要）
   > ⚠️ **只有linux版本**
   > ⚠️ **CUDA 兼容性提示**
   > Mamba 预编译包与 **CUDA 版本强相关**。
   > 请确保安装与本地 CUDA 版本匹配的包（如 `cu11` 或 `cu12`）。
   > 版本不匹配可能导致运行时错误或导入失败。

   **CUDA 12** 示例：

   ```bash
   pip install https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
   ```

5. 安装 Moirai 模型依赖（models/Moirai.py 需要）
   ```bash
   pip install uni2ts --no-deps
   ```

### Docker 部署

```bash
# 构建并以后台模式启动容器
docker compose -f 'Time-Series-Library/docker-compose.yml' up -d --build

# 在仓库根目录创建 ./dataset 并下载/放置数据集
mkdir -p dataset

# 将本地数据集复制到容器内 /workspace/dataset
docker cp ./dataset tslib:/workspace/dataset

# 进入运行中的容器
docker exec -it tslib bash

# 切换到容器内的工作目录
cd /workspace

# 使用预训练 Moirai 模型进行零样本预测
python -u run.py \
  --task_name zero_shot_forecast \   # 任务类型：零样本预测
  --is_training 0 \                  # 0 = 仅推理
  --root_path ./dataset/ETT-small/ \ # 数据集根路径
  --data_path ETTh1.csv \            # 数据文件名
  --model_id ETTh1_512_96 \          # 实验/模型标识
  --model Moirai \                   # 模型名称（TimesFM / Moirai）
  --data ETTh1 \                     # 数据集名称
  --features M \                     # 多变量预测
  --seq_len 512 \                    # 输入序列长度
  --pred_len 96 \                    # 预测步长
  --enc_in 7 \                       # 输入变量数
  --des 'Exp' \                      # 实验描述
  --itr 1                             # 运行次数
```

### 快速测试

5个任务快速测试（每个任务1个epoch）：

```bash
# 执行所有5个任务的快速测试
export CUDA_VISIBLE_DEVICES=0

# 1. 长期预测
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_long --model DLinear --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --train_epochs 1 --num_workers 2

# 2. 短期预测（使用ETT数据集，较短预测长度）
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_short --model TimesNet --data ETTh1 --features M --seq_len 24 --label_len 12 --pred_len 24 --e_layers 2 --d_layers 1 --d_model 16 --d_ff 32 --enc_in 7 --dec_in 7 --c_out 7 --top_k 5 --train_epochs 1 --num_workers 2

# 3. 插补
python -u run.py --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_imp --model TimesNet --data ETTh1 --features M --seq_len 96 --e_layers 2 --d_layers 1 --d_model 16 --d_ff 32 --enc_in 7 --dec_in 7 --c_out 7 --top_k 3 --train_epochs 1 --num_workers 2 --label_len 0 --pred_len 0 --mask_rate 0.125 --learning_rate 0.001

# 4. 异常检测
python -u run.py --task_name anomaly_detection --is_training 1 --root_path ./dataset/PSM --model_id test_ad --model TimesNet --data PSM --features M --seq_len 100 --pred_len 0 --d_model 64 --d_ff 64 --e_layers 2 --enc_in 25 --c_out 25 --anomaly_ratio 1.0 --top_k 3 --train_epochs 1 --batch_size 128 --num_workers 2

# 5. 分类
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/Heartbeat/ --model_id Heartbeat --model TimesNet --data UEA --e_layers 2 --d_layers 1 --factor 3 --d_model 64 --d_ff 128 --top_k 3 --train_epochs 1 --batch_size 16 --learning_rate 0.001 --num_workers 0
```

### 训练与评测

`./scripts/` 目录下提供了全部基准的实验脚本，可参考下列示例复现实验：

> ⚠️ 部分脚本中默认设置了 `CUDA_VISIBLE_DEVICES`，请根据实际 GPU 配置修改或删除该设置，否则可能导致无法使用 GPU。

```bash
# 长期预测
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# 短期预测
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# 插补
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# 异常检测
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# 分类
bash ./scripts/classification/TimesNet.sh
```


### 开发自定义模型
- 将模型文件放入 `./models`，可参考 `./models/Transformer.py`。
- 在 `./scripts` 下创建对应的运行脚本。

### 注意事项：

(1) 关于分类：由于我们在统一代码库中涵盖五大任务，各子任务的精度可能略有波动，但平均性能可复现（甚至略高）。复现用 checkpoint 可在 [此处](https://github.com/thuml/Time-Series-Library/issues/494) 下载。

(2) 关于异常检测：有关异常检测调整策略的讨论见[这里](https://github.com/thuml/Anomaly-Transformer/issues/14)，核心是该调整策略对应事件级指标。

### 查看项目文件结构：

```
Time-Series-Library/
├── README.md                     # 官方README，包含任务、榜单、使用方法
├── requirements.txt              # pip依赖列表，直接pip install复现环境
├── LICENSE / CONTRIBUTING.md     # 原项目许可与贡献指南
├── run.py                        # 单入口脚本，解析参数并调度各任务
├── exp/                          # 各任务实验管线，封装训练/验证/测试
│   ├── exp_basic.py              # 实验基类，注册所有模型，统一构建流程
│   ├── exp_long_term_forecasting.py    # 长期预测实验逻辑
│   ├── exp_short_term_forecasting.py   # 短期预测实验逻辑
│   ├── exp_imputation.py               # 缺失值填充实验
│   ├── exp_anomaly_detection.py        # 异常检测实验
│   ├── exp_classification.py           # 分类实验
│   └── exp_zero_shot_forecasting.py    # LTSM零样本预测评估
├── data_provider/                # 数据入口，负责数据集载入与切分
│   ├── data_factory.py           # 根据任务选择对应DataLoader
│   ├── data_loader.py            # 通用时序数据读取与滑窗逻辑
│   ├── uea.py / m4.py            # UEA、M4等特定数据格式处理
│   └── __init__.py               # 暴露上层可用的数据工厂接口
├── models/                       # 所有模型实现，文件名即模型名
│   ├── TimesNet.py、TimeMixer.py 等 # 主流预测模型
│   ├── Chronos2.py、TiRex.py     # LTSM零样本模型
│   └── __init__.py               # 统一导出供实验模块按名称实例化
├── layers/                       # 复用层/块，如注意力、卷积、嵌入
│   ├── Transformer_EncDec.py     # Transformer编解码堆栈
│   ├── AutoCorrelation.py        # 自相关算子
│   ├── MultiWaveletCorrelation.py# 频域单元
│   └── Embed.py 等               # 各模型共享基元
├── utils/                        # 工具集合
│   ├── metrics.py                # MSE/MAE/DTW等评估指标
│   ├── tools.py                  # 训练通用工具，比如EarlyStopping
│   ├── augmentation.py           # 分类/检测任务增强策略
│   ├── print_args.py             # 统一打印参数
│   └── masking.py / losses.py    # 任务相关辅助函数
├── scripts/                      # 复现实验的bash脚本
│   ├── long_term_forecast/       # 按数据集/模型划分的长期预测脚本
│   ├── short_term_forecast/      # M4等短期预测脚本
│   ├── imputation/               # 多数据集缺失填充脚本
│   ├── anomaly_detection/        # SMD/SMAP/SWAT等检测脚本
│   ├── classification/           # UEA分类脚本
│   └── exogenous_forecast/       # TimeXer外生变量预测流程
├── tutorial/                     # 官方TimesNet教学notebook与插图
└── pic/                          # README插图（数据集分布等）
```

### 理解项目架构：

- **整体流程**：通过 `scripts/*.sh` 设定实验参数 → 调用 `python run.py ...` → `run.py` 解析参数并根据 `task_name` 选择对应 `Exp_*` 类 → `Exp_*` 内部利用 `data_provider` 构造数据加载器、`models` 实例化网络、`utils` 中的工具完成训练/验证/测试 → 结果与模型参数写入 `./checkpoints`。
- **实验层（exp/）**：`Exp_Basic` 负责注册模型与设备，子类实现 `_get_data/train/test`，将不同任务的差异隔离，方便模型在多任务间复用。
- **模型与层（models/ + layers/）**：模型文件集中定义各网络结构，公用的注意力、卷积、频域块等沉淀在 `layers/`，减少重复实现。
- **数据层（data_provider/）**：`data_factory` 按任务返回 Dataset/DataLoader，`data_loader` 封装序列裁剪、滑动窗口、掩码策略，不同任务通过参数控制窗口长度、缺失率、异常比例。
- **脚本层（scripts/）**：提供与论文一致的复现实验脚本，涵盖各种数据集/模型/GPU 配置，便于批量跑榜，也可作为自定义实验的起点。
- **辅助层（utils/）**：`metrics` 统一评估指标，`tools` 中的 `EarlyStopping`、`adjust_learning_rate` 等负责训练调度；`augmentation`/`masking` 等用于任务特定的数据增强或预处理。
- **学习建议**：阅读顺序推荐 `scripts -> run.py -> exp/exp_basic.py -> 对应 Exp 子类 -> data_provider -> models`，并结合 `tutorial/TimesNet_tutorial.ipynb` 快速熟悉整体调用链，再按需深入模型或层级实现。

## 引用

如果本仓库对您有帮助，请引用以下论文：

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}

@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```

## 联系方式
如有问题或建议，欢迎联系维护团队：

现任：
- Haixu Wu（博士，wuhaixu98@gmail.com）
- Yuxuan Wang（博士生，wangyuxu22@mails.tsinghua.edu.cn）
- Yong Liu（博士生，liuyong21@mails.tsinghua.edu.cn）
- Ailuntz（开源社区学生，ailuntz@icloud.com）

往届：
- Huikun Weng（本科生，wenghk22@mails.tsinghua.edu.cn）
- Tengge Hu（硕士，htg21@mails.tsinghua.edu.cn）
- Haoran Zhang（硕士，z-hr20@mails.tsinghua.edu.cn）
- Jiawei Guo（本科生，guo-jw21@mails.tsinghua.edu.cn）

也欢迎在 Issues 中反馈。

## 致谢

本库参考了以下仓库：

- 预测：https://github.com/thuml/Autoformer
- 异常检测：https://github.com/thuml/Anomaly-Transformer
- 分类：https://github.com/thuml/Flowformer

实验所用数据集均为公开数据，来源如下：

- 长期预测与插补：https://github.com/thuml/Autoformer
- 短期预测：https://github.com/ServiceNow/N-BEATS
- 异常检测：https://github.com/thuml/Anomaly-Transformer
- 分类：https://www.timeseriesclassification.com/

## 感谢所有贡献者

<a href="https://github.com/thuml/Time-Series-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=thuml/Time-Series-Library" />
</a>
