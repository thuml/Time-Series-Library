# Time Series Library (TSLib)
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.

> **中文文档**：[README_zh.md](./README_zh.md)

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting, imputation, anomaly detection, and classification.**

:triangular_flag_on_post:**News** (2025.12) Many thanks to the great work from [ailuntz](https://github.com/thuml/Time-Series-Library/pull/805), which provides an updated requirements and docker deployment, as well as a well-organized document. This is quite meaningful to this project and beginners.

:triangular_flag_on_post:**News** (2025.11) Considering the rapid development of Large Time Series Models (LTSMs), we have newly added a [[zero-shot forecasting]](https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_zero_shot_forecasting.py) feature in TSLib. You can try [this script](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/LTSM.sh) to evaluate LTSMs.

:triangular_flag_on_post:**News** (2025.10) Given the recent confusion among researchers regarding minor improvements on standard benchmarks, we propose the [[Accuracy Law]](https://arxiv.org/abs/2510.02729) to characterize the objectives of deep time series forecasting tasks, which can be used to identify saturated datasets.

:triangular_flag_on_post:**News** (2024.10) We have included [[TimeXer]](https://arxiv.org/abs/2402.19072), which defined a practical forecasting paradigm: Forecasting with Exogenous Variables. Considering both practicability and computation efficiency, we believe the new forecasting paradigm defined in TimeXer can be the "right" task for future research.

:triangular_flag_on_post:**News** (2024.10) Our lab has open-sourced [[OpenLTM]](https://github.com/thuml/OpenLTM), which provides a distinct pretrain-finetuning paradigm compared to TSLib. If you are interested in Large Time Series Models, you may find this repository helpful.

:triangular_flag_on_post:**News** (2024.07) We wrote a comprehensive survey of [[Deep Time Series Models]](https://arxiv.org/abs/2407.13278) with a rigorous benchmark based on TSLib. In this paper, we summarized the design principles of current time series models supported by insightful experiments, hoping to be helpful to future research.

:triangular_flag_on_post:**News** (2024.04) Many thanks for the great work from [frecklebars](https://github.com/thuml/Time-Series-Library/pull/378). The famous sequential model [Mamba](https://arxiv.org/abs/2312.00752) has been included in our library. See [this file](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py), where you need to install `mamba_ssm` with pip at first.

:triangular_flag_on_post:**News** (2024.03) Given the inconsistent look-back length of various papers, we split the long-term forecasting in the leaderboard into two categories: Look-Back-96 and Look-Back-Searching. We recommend researchers read [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2), which includes both look-back length settings in experiments for scientific rigor.

:triangular_flag_on_post:**News** (2023.10) We add an implementation to [iTransformer](https://arxiv.org/abs/2310.06625), which is the state-of-the-art model for long-term forecasting. The official code and complete scripts of iTransformer can be found [here](https://github.com/thuml/iTransformer).

:triangular_flag_on_post:**News** (2023.09) We added a detailed [tutorial](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb) for [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) and this library, which is quite friendly to beginners of deep time series analysis.

:triangular_flag_on_post:**News** (2023.02) We release the TSlib as a comprehensive benchmark and code base for time series models, which is extended from our previous GitHub repository [Autoformer](https://github.com/thuml/Autoformer).

## Leaderboard for Time Series Analysis

Till March 2024, the top three models for five different tasks are:

| Model<br>Ranking | Long-term<br>Forecasting<br>Look-Back-96              | Long-term<br/>Forecasting<br/>Look-Back-Searching     | Short-term<br>Forecasting                                    | Imputation                                                   | Classification                                               | Anomaly<br>Detection                               |
| ---------------- | ----------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| 🥇 1st            | [TimeXer](https://arxiv.org/abs/2402.19072)      | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2) | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)       |
| 🥈 2nd            | [iTransformer](https://arxiv.org/abs/2310.06625) | [PatchTST](https://github.com/yuqinie98/PatchTST)     | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 🥉 3rd            | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2)          | [DLinear](https://arxiv.org/pdf/2205.13504.pdf)       | [FEDformer](https://github.com/MAZiqing/FEDformer)           | [Autoformer](https://github.com/thuml/Autoformer)            | [Informer](https://github.com/zhouhaoyi/Informer2020)        | [Autoformer](https://github.com/thuml/Autoformer)  |


**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.
  - [x] **TimeXer** - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[NeurIPS 2024]](https://arxiv.org/abs/2402.19072) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py)
  - [x] **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py).
  - [x] **TSMixer** - TSMixer: An All-MLP Architecture for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/pdf/2303.06053.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py)
  - [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py).
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py).
  - [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py).
  - [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py).
  - [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py).
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py).
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py).
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py).
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py).
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py).

See our latest paper [[TimesNet]](https://arxiv.org/abs/2210.02186) for the comprehensive benchmark. We will release a real-time updated online version soon.

**Newly added baselines.** We will add them to the leaderboard after a comprehensive evaluation.
  - [x] **TimeFilter** - TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting [[ICML 2025]](https://arxiv.org/abs/2501.13041) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeFilter.py)
  - [x] **KAN-AD** - KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks [[ICML 2025]](https://arxiv.org/abs/2411.00278) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/KANAD.py)
  - [x] **MultiPatchFormer** - A multiscale model for multivariate time series forecasting [[Scientific Reports 2025]](https://www.nature.com/articles/s41598-024-82417-4) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MultiPatchFormer.py)
  - [x] **WPMixer** - WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting [[AAAI 2025]](https://arxiv.org/abs/2412.17176) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/WPMixer.py)
  - [x] **MSGNet** - MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting [[AAAI 2024]](https://dl.acm.org/doi/10.1609/aaai.v38i10.28991) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MSGNet.py)
  - [x] **PAttn** - Are Language Models Actually Useful for Time Series Forecasting? [[NeurIPS 2024]](https://arxiv.org/pdf/2406.16964) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PAttn.py)
  - [x] **Mamba** - Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[arXiv 2023]](https://arxiv.org/abs/2312.00752) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py)
  - [x] **SegRNN** - SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [[arXiv 2023]](https://arxiv.org/abs/2308.11200.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SegRNN.py).
  - [x] **Koopa** - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [[NeurIPS 2023]](https://arxiv.org/pdf/2305.18803.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Koopa.py).
  - [x] **FreTS** - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [[NeurIPS 2023]](https://arxiv.org/pdf/2311.06184.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FreTS.py).
  - [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py).
  - [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py).
  - [x] **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py).
  - [x] **SCINet** - SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction [[NeurIPS 2022]](https://openreview.net/pdf?id=AyajSjTAzmg)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SCINet.py).
  - [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py).
  - [x] **TFT** - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting [[arXiv 2019]](https://arxiv.org/abs/1912.09363)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TemporalFusionTransformer.py).

**Newly added Large Time Series Models.** This library also supports the zero-shot evaluation of the following LTSMs.

- [x] **Chronos2** - Chronos-2: From Univariate to Universal Forecasting [[arXiv 2025]](https://arxiv.org/abs/2510.15821) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Chronos2.py)
- [x] **TiRex** - TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning [[NeurIPS 2025]](https://arxiv.org/pdf/2505.23719) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiRex.py)
- [x] **Sundial** - Sundial: A Family of Highly Capable Time Series Foundation Models [[ICML 2025]](https://arxiv.org/pdf/2502.00816) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Sundial.py)
- [x] **Time-MoE** - Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts [[ICLR 2025]](https://arxiv.org/pdf/2409.16040) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMoE.py)
- [x] **Toto** - Toto: Time Series Optimized Transformer for Observability [arXiv 2024](https://arxiv.org/pdf/2407.07874)
- [x] **Chronos** - Chronos: Learning the Language of Time Series [[TMLR 2024]](https://arxiv.org/pdf/2403.07815) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Chronos.py)
- [x] **Moirai** - Unified Training of Universal Time Series Forecasting Transformers [[ICML 2024]](https://arxiv.org/pdf/2402.02592)
- [x] **TimesFM** - A decoder-only foundation model for time-series forecasting [[ICML 2024]](https://arxiv.org/abs/2310.10688) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesFM.py)


 
## Getting Started

### Prepare Data
You can obtain the well-preprocessed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) or [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library). Then place the downloaded data in the folder `./dataset`.

### Installation
1. Clone this repository.
   ```bash
   git clone https://github.com/thuml/Time-Series-Library.git
   cd Time-Series-Library
   ```

2. Create a new Conda environment.
   ```bash
   conda create -n tslib python=3.11
   conda activate tslib
   ```

3. Install Core Dependencies
   > ⚠️ **CUDA Compatibility Notice**
   > The torch prebuilt package is **CUDA-version specific**. (See https://pytorch.org/get-started/previous-versions/)
   > Please make sure to install the package that matches your local CUDA version (e.g., `cu118` or `cu121`).
   > Recommended: torch==2.5.1

   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

   pip install -r requirements.txt
   ```

4. Install Dependencies for Mamba Model (Required for Time-Series-Library/models/Mamba.py)
   > ⚠️ **Linux only**
   > ⚠️ **CUDA Compatibility Notice**
   > The prebuilt Mamba wheel is **CUDA-version specific**.
   > Please make sure to install the wheel that matches your local CUDA version
   > (e.g., `cu11` or `cu12`). Installing a mismatched version may result in
   > runtime errors or import failures.

   Example for **CUDA 12**:

   ```bash
   pip install https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
   ```

5. Install Dependencies for Moirai Model (Required for Time-Series-Library/models/Moirai.py)
   ```bash
   pip install uni2ts --no-deps
   ```

### Docker Deployment
```bash
# Build and start the Docker container in detached mode
docker compose -f 'Time-Series-Library/docker-compose.yml' up -d --build

# Download / place the dataset into a newly created folder ./dataset at the repository root
mkdir -p dataset  # create the dataset directory

# Copy the local dataset into the container at /workspace/dataset
docker cp ./dataset tslib:/workspace/dataset

# Enter the running container to continue training / evaluation
docker exec -it tslib bash

# Switch to the workspace directory inside the container
cd /workspace

# Run zero-shot forecasting with the pre-trained Moirai model
python -u run.py \
  --task_name zero_shot_forecast \   # task type: zero-shot forecasting
  --is_training 0 \                  # 0 = inference only (no training)
  --root_path ./dataset/ETT-small/ \ # root directory of the dataset
  --data_path ETTh1.csv \            # dataset file name
  --model_id ETTh1_512_96 \          # experiment/model identifier
  --model Moirai \                   # model name (TimesFM / Moirai)
  --data ETTh1 \                     # dataset name
  --features M \                     # multivariate forecasting
  --seq_len 512 \                    # input sequence length
  --pred_len 96 \                    # prediction horizon
  --enc_in 7 \                       # number of input variables
  --des 'Exp' \                      # experiment description
  --itr 1                             # number of runs
```


### Quick Test

Quick test for all 5 tasks (1 epoch each):

```bash
# Run quick tests for all 5 tasks
export CUDA_VISIBLE_DEVICES=0

# 1. Long-term forecasting
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_long --model DLinear --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --train_epochs 1 --num_workers 2

# 2. Short-term forecasting (using ETT dataset with shorter prediction length)
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_short --model TimesNet --data ETTh1 --features M --seq_len 24 --label_len 12 --pred_len 24 --e_layers 2 --d_layers 1 --d_model 16 --d_ff 32 --enc_in 7 --dec_in 7 --c_out 7 --top_k 5 --train_epochs 1 --num_workers 2

# 3. Imputation
python -u run.py --task_name imputation --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id test_imp --model TimesNet --data ETTh1 --features M --seq_len 96 --e_layers 2 --d_layers 1 --d_model 16 --d_ff 32 --enc_in 7 --dec_in 7 --c_out 7 --top_k 3 --train_epochs 1 --num_workers 2 --label_len 0 --pred_len 0 --mask_rate 0.125 --learning_rate 0.001

# 4. Anomaly detection
python -u run.py --task_name anomaly_detection --is_training 1 --root_path ./dataset/PSM --model_id test_ad --model TimesNet --data PSM --features M --seq_len 100 --pred_len 0 --d_model 64 --d_ff 64 --e_layers 2 --enc_in 25 --c_out 25 --anomaly_ratio 1.0 --top_k 3 --train_epochs 1 --batch_size 128 --num_workers 2

# 5. Classification
python -u run.py --task_name classification --is_training 1 --root_path ./dataset/Heartbeat/ --model_id Heartbeat --model TimesNet --data UEA --e_layers 2 --d_layers 1 --factor 3 --d_model 64 --d_ff 128 --top_k 3 --train_epochs 1 --batch_size 16 --learning_rate 0.001 --num_workers 0
```


### Train and Evaluate

We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

> ⚠️ Some scripts have `CUDA_VISIBLE_DEVICES` set by default. Please modify or remove this setting according to your actual GPU configuration, otherwise it may prevent GPU usage.

```bash
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

### Develop Your Own Model
- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Create the corresponding scripts under the folder `./scripts`.

### Note: 

(1) About classification: Since we include all five tasks in a unified code base, the accuracy of each subtask may fluctuate but the average performance can be reproduced (even a bit better). We have provided the reproduced checkpoints [here](https://github.com/thuml/Time-Series-Library/issues/494).

(2) About anomaly detection: Some discussion about the adjustment strategy in anomaly detection can be found [here](https://github.com/thuml/Anomaly-Transformer/issues/14). The key point is that the adjustment strategy corresponds to an event-level metric.

### Inspect the project structure:

```
Time-Series-Library/
├── README.md                     # Official README with tasks, leaderboard, usage
├── requirements.txt              # pip dependency list for quick environment setup
├── LICENSE / CONTRIBUTING.md     # Upstream license and contribution guide
├── run.py                        # Unified entry that parses args and dispatches tasks
├── exp/                          # Task pipelines wrapping train/val/test
│   ├── exp_basic.py              # Experiment base class, registers models, builds flows
│   ├── exp_long_term_forecasting.py    # Long-term forecasting logic
│   ├── exp_short_term_forecasting.py   # Short-term forecasting logic
│   ├── exp_imputation.py               # Missing-value imputation
│   ├── exp_anomaly_detection.py        # Anomaly detection
│   ├── exp_classification.py           # Classification
│   └── exp_zero_shot_forecasting.py    # LTSM zero-shot evaluation
├── data_provider/                # Dataset loaders and splits
│   ├── data_factory.py           # Chooses the proper DataLoader per task
│   ├── data_loader.py            # Generic TS reader with sliding-window logic
│   ├── uea.py / m4.py            # Parsers for UEA, M4 and other formats
│   └── __init__.py               # Exposes factory interfaces upward
├── models/                       # All model implementations
│   ├── TimesNet.py, TimeMixer.py # Main forecasting models
│   ├── Chronos2.py, TiRex.py     # LTSM zero-shot models
│   └── __init__.py               # Enables name-based instantiation inside exp
├── layers/                       # Reusable attention / conv / embedding blocks
│   ├── Transformer_EncDec.py     # Transformer stacks
│   ├── AutoCorrelation.py        # Auto-correlation operator
│   ├── MultiWaveletCorrelation.py# Frequency-domain unit
│   └── Embed.py etc.             # Shared primitives
├── utils/                        # Utility toolbox
│   ├── metrics.py                # MSE / MAE / DTW and other metrics
│   ├── tools.py                  # General helpers such as EarlyStopping
│   ├── augmentation.py           # Augmentations for classification / detection
│   ├── print_args.py             # Unified argument printer
│   └── masking.py / losses.py    # Task-specific helpers
├── scripts/                      # Bash recipes for reproducible experiments
│   ├── long_term_forecast/       # Long-term forecasting per dataset/model
│   ├── short_term_forecast/      # M4 and other short-term scripts
│   ├── imputation/               # Imputation scripts
│   ├── anomaly_detection/        # SMD / SMAP / SWAT detection scripts
│   ├── classification/           # UEA classification scripts
│   └── exogenous_forecast/       # TimeXer exogenous forecasting flow
├── tutorial/                     # TimesNet tutorial notebook and figures
└── pic/                          # README figures (dataset overview, etc.)
```

### Understand the project architecture:

- **E2E flow**: configure experiments via `scripts/*.sh` → run `python run.py ...` → `run.py` parses arguments and selects the proper `Exp_*` via `task_name` → the experiment builds datasets through `data_provider`, instantiates networks from `models`, and drives train/val/test with utilities in `utils` → metrics and checkpoints are written to `./checkpoints`.
- **Experiment layer (`exp/`)**: `Exp_Basic` registers models and devices; subclasses implement `_get_data`, `train`, and `test` to encapsulate task-specific differences so the same model can be reused.
- **Model & layer layer (`models/` + `layers/`)**: model files define architectures, while reusable attention/conv/frequency components live in `layers/` to minimize duplication.
- **Data layer (`data_provider/`)**: `data_factory` returns the correct `Dataset/DataLoader`; `data_loader` handles windowing, masking, and sampling, with arguments controlling window length, missing ratio, anomaly ratio, etc.
- **Script layer (`scripts/`)**: bash scripts capture paper configurations (dataset, window, model, GPU) for reproducibility and serve as templates for custom runs.
- **Utility layer (`utils/`)**: `metrics` centralizes evaluation, `tools` bundles essentials like `EarlyStopping` and `adjust_learning_rate`, while `augmentation`/`masking` cover task-specific preprocessing.
- **Learning path**: recommended reading order is `scripts -> run.py -> exp/exp_basic.py -> corresponding Exp subclass -> data_provider -> models`, using `tutorial/TimesNet_tutorial.ipynb` as a guided walkthrough before diving deeper.

## Citation

If you find this repo useful, please cite our paper.

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

## Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

Current:
- Haixu Wu (Ph.D., wuhaixu98@gmail.com)
- Yuxuan Wang (Ph.D. student, wangyuxu22@mails.tsinghua.edu.cn)
- Yong Liu (Ph.D. student, liuyong21@mails.tsinghua.edu.cn)
- Ailuntz (Student from Open-source Community, ailuntz@icloud.com)

Previous:
- Huikun Weng (Undergraduate, wenghk22@mails.tsinghua.edu.cn)
- Tengge Hu (Master student, htg21@mails.tsinghua.edu.cn)
- Haoran Zhang (Master student, z-hr20@mails.tsinghua.edu.cn)
- Jiawei Guo (Undergraduate, guo-jw21@mails.tsinghua.edu.cn)

Or describe it in Issues.

## Acknowledgement

This library is constructed based on the following repos:

- Forecasting: https://github.com/thuml/Autoformer.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://github.com/thuml/Flowformer.

All the experiment datasets are public, and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer.

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://www.timeseriesclassification.com/.

## All Thanks To Our Contributors

<a href="https://github.com/thuml/Time-Series-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=thuml/Time-Series-Library" />
</a>
