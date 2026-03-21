# TimesNet 论文复现指南

这份文档对应仓库里的 `article/TimesNet.pdf`，目标是把仓库现有的 TimesNet 实验脚本真正跑通，并把结果按论文口径汇总出来，方便你判断“有没有复现到论文水平”。

## 1. 先明确复现目标

`TimesNet.pdf` 不是只做了一个任务，而是覆盖了 5 类任务：

- 长期预测 `long_term_forecast`
- 短期预测 `short_term_forecast`
- 缺失值填补 `imputation`
- 分类 `classification`
- 异常检测 `anomaly_detection`

所以“复现 TimesNet 论文”通常不是跑一个命令，而是按这 5 类任务分别跑对应脚本，再把结果和论文表格对齐。

## 2. 你现在应该按什么顺序做

推荐顺序如下：

1. 先确认环境和数据都正常。
2. 先做一个很小的冒烟测试，确保代码、数据、GPU、依赖都通。
3. 先跑长期预测，因为这是最标准、最容易判断是否跑通的主线任务。
4. 再跑短期预测、缺失值填补、分类、异常检测。
5. 最后统一汇总结果，和论文表格做对照。

这样做的原因很简单：

- 如果你一上来直接跑全量脚本，一旦环境、GPU、数据路径有问题，会浪费很长时间。
- 长期预测的输出最直观，能最快验证训练和测试流程是否正常。
- 论文主表里很多结果是“平均值”，不是单次运行日志里的某一个值，所以最后必须做一次汇总。

## 3. 这次我给你补了什么

为了避免你直接跑原始脚本时踩坑，我新增了两个工具：

- `scripts/reproduce_timesnet.sh`
  作用：统一调用仓库原始 TimesNet 脚本，并自动绕开原脚本里写死的 `CUDA_VISIBLE_DEVICES=2/4/5/7`。
- `scripts/summarize_timesnet_results.py`
  作用：把你跑出来的结果自动聚合成一个对照报告 `TIMESNET_REPRODUCTION_RESULTS.md`。

为什么要这样做：

- 这台机器实际只有 `GPU 0` 可见。
- 原始脚本里很多 `CUDA_VISIBLE_DEVICES` 写成了 `2`、`4`、`5`、`7`，直接跑很容易把 GPU 配错。
- 分类和异常检测日志里输出的是 `0~1` 的比例值，但论文表格写的是百分数，这一步手工对很容易看错。

## 4. 环境和数据先检查什么

你已经指定使用 Anaconda 的 `timesnet` 环境，所以默认直接用它。

先检查：

```bash
conda run -n timesnet python --version
conda run -n timesnet python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

再检查数据目录：

```bash
ls dataset/ETT-small
ls dataset/m4
ls dataset/electricity
ls dataset/weather
ls dataset/traffic
ls dataset/exchange_rate
ls dataset/illness
ls dataset/PSM
ls dataset/SMD
ls dataset/SMAP
ls dataset/MSL
ls dataset/SWaT
ls dataset/EthanolConcentration
```

你这个仓库里这些公开数据已经在 `dataset/` 下，基础条件是满足的。

## 5. 第一步不要直接跑全量，先做冒烟测试

先用一个 1 epoch 的小命令确认训练入口正常：

```bash
conda run --no-capture-output -n timesnet python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id smoke_ETTh1_96_96 \
  --model TimesNet \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --train_epochs 1 \
  --patience 1 \
  --num_workers 0 \
  --gpu 0 \
  --des smoke \
  --itr 1
```

如果这个命令能正常训练并打印出类似 `mse:... mae:...`，说明下面的全量复现可以开始了。

## 6. 正式复现时用哪个脚本

统一入口：

```bash
bash scripts/reproduce_timesnet.sh <task> [target]
```

支持的任务：

- `long_term`
- `short_term`
- `imputation`
- `classification`
- `anomaly`
- `all`

### 6.1 长期预测

先跑长期预测最合适。

跑全部长期预测数据集：

```bash
bash scripts/reproduce_timesnet.sh long_term all
```

如果你只想先跑一个数据集：

```bash
bash scripts/reproduce_timesnet.sh long_term etth1
bash scripts/reproduce_timesnet.sh long_term ettm1
bash scripts/reproduce_timesnet.sh long_term traffic
```

这一步对应论文主文 Table 2，仓库脚本对应论文 Appendix Table 13 的完整结果。

### 6.2 短期预测

M4 是单独一套流程：

```bash
bash scripts/reproduce_timesnet.sh short_term
```

它会依次跑：

- Yearly
- Quarterly
- Monthly
- Weekly
- Daily
- Hourly

全部 6 个频率跑完以后，才能按论文口径计算加权平均 `SMAPE / MASE / OWA`。

### 6.3 缺失值填补

跑全部：

```bash
bash scripts/reproduce_timesnet.sh imputation all
```

只跑一个：

```bash
bash scripts/reproduce_timesnet.sh imputation etth1
bash scripts/reproduce_timesnet.sh imputation weather
```

这一步对应论文主文 Table 4，仓库脚本对应论文 Appendix Table 16 的完整结果。

### 6.4 分类

分类脚本会一次跑完 10 个 UEA 子数据集：

```bash
bash scripts/reproduce_timesnet.sh classification
```

这一步对应论文 Figure 5 和 Appendix Table 17。

### 6.5 异常检测

跑全部：

```bash
bash scripts/reproduce_timesnet.sh anomaly all
```

只跑一个：

```bash
bash scripts/reproduce_timesnet.sh anomaly psm
bash scripts/reproduce_timesnet.sh anomaly smd
```

注意：

- `SWaT` 脚本不是只跑一个配置，而是连续试多个配置。
- 最后和论文比的时候，应该按数据集取最优 F1，而不是只看最后一条日志。

## 7. 每一步跑完以后结果会落到哪里

主要看这几个目录和文件：

- `checkpoints/`
  训练好的模型权重。
- `results/`
  主要任务的 `metrics.npy`、分类结果文件等。
- `test_results/`
  一些可视化图和测试输出。
- `m4_results/TimesNet/`
  M4 预测结果 CSV。
- `result_long_term_forecast.txt`
  长期预测日志摘要。
- `result_imputation.txt`
  缺失值填补日志摘要。
- `result_anomaly_detection.txt`
  异常检测日志摘要。

## 8. 跑完以后怎么和论文对表

不要人工一条条翻日志，直接运行汇总脚本：

```bash
conda run --no-capture-output -n timesnet python scripts/summarize_timesnet_results.py
```

运行后会生成：

```bash
TIMESNET_REPRODUCTION_RESULTS.md
```

这个文件会自动做几件事：

- 把长期预测按 4 个预测长度做平均，再和论文 Table 2 对照。
- 把缺失值填补按 4 个 mask ratio 做平均，再和论文 Table 4 对照。
- 从 `m4_results/TimesNet/` 重新计算 `SMAPE / MASE / OWA`，再和论文 Table 3 对照。
- 把分类任务从 `0~1` 精度转换成百分数，再和论文 Table 17 对照。
- 把异常检测从 `0~1` F1 转成百分数，并且同一数据集取最优 F1，再和论文 Table 5 对照。

## 9. 论文里应该对齐到哪些数

下面这些是你最终最应该盯住的主表目标值。

来源：

- 论文 arXiv: https://arxiv.org/abs/2210.02186
- 可读 HTML 版: https://ar5iv.labs.arxiv.org/html/2210.02186

### 9.1 长期预测 Table 2

这些数是 4 个预测长度的平均值：

| Dataset | Paper MSE | Paper MAE |
| --- | --- | --- |
| ETTm1 | 0.400 | 0.406 |
| ETTm2 | 0.291 | 0.333 |
| ETTh1 | 0.458 | 0.450 |
| ETTh2 | 0.414 | 0.427 |
| Electricity | 0.192 | 0.295 |
| Traffic | 0.620 | 0.336 |
| Weather | 0.259 | 0.287 |
| Exchange | 0.416 | 0.443 |
| ILI | 2.139 | 0.931 |

### 9.2 短期预测 Table 3

M4 加权平均目标值：

| Metric | Paper |
| --- | --- |
| SMAPE | 11.829 |
| MASE | 1.585 |
| OWA | 0.851 |

### 9.3 缺失值填补 Table 4

这些数是 4 个 mask ratio 的平均值：

| Dataset | Paper MSE | Paper MAE |
| --- | --- | --- |
| ETTm1 | 0.027 | 0.107 |
| ETTm2 | 0.022 | 0.088 |
| ETTh1 | 0.078 | 0.187 |
| ETTh2 | 0.049 | 0.146 |
| Electricity | 0.092 | 0.210 |
| Weather | 0.030 | 0.054 |

### 9.4 分类

论文主文给出的平均准确率目标值：

| Metric | Paper |
| --- | --- |
| Average Accuracy (%) | 73.6 |

更细的 10 个子数据集精度，见论文 Appendix Table 17。汇总脚本已经把这些目标值内置进去了。

### 9.5 异常检测 Table 5

这里应该对照的是仓库当前 Inception 版 TimesNet 的 F1：

| Dataset | Paper F1 (%) |
| --- | --- |
| SMD | 85.12 |
| MSL | 84.18 |
| SMAP | 70.85 |
| SWAT | 92.10 |
| PSM | 95.21 |
| Average | 85.49 |

## 10. 你最容易踩的坑

### 10.1 直接跑原始脚本

原始脚本很多写死了：

```bash
export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=4
export CUDA_VISIBLE_DEVICES=5
export CUDA_VISIBLE_DEVICES=7
```

如果你的机器没有这些卡号，脚本就会跑错设备，甚至直接退回 CPU 或失败。

所以建议直接使用：

```bash
bash scripts/reproduce_timesnet.sh ...
```

### 10.2 只看一条日志，不做平均

论文主表很多不是某一个单次实验值，而是：

- 长期预测：4 个预测长度平均
- 缺失值填补：4 个 mask ratio 平均
- 短期预测：M4 6 个频率加权汇总
- 分类：10 个 UEA 子集平均
- 异常检测：5 个数据集平均

所以一定要跑汇总脚本，而不是只截某一行日志。

### 10.3 分类和异常检测的单位看错

仓库日志里通常打印：

- `accuracy:0.357`
- `F-score : 0.952`

论文里对应的是：

- `35.7%`
- `95.2%`

汇总脚本已经做了自动换算。

### 10.4 SWaT 不是单配置

`scripts/anomaly_detection/SWAT/TimesNet.sh` 会连续尝试多组超参。

所以：

- 你不能只看最后一条。
- 应该按同一数据集取最优 F1 再和论文比。

## 11. 我建议你的实际执行顺序

如果你是第一次复现，最稳妥的执行顺序就是这 6 条：

```bash
conda run --no-capture-output -n timesnet python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id smoke_ETTh1_96_96 --model TimesNet --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 16 --d_ff 32 --top_k 5 --train_epochs 1 --patience 1 --num_workers 0 --gpu 0 --des smoke --itr 1

bash scripts/reproduce_timesnet.sh long_term all
bash scripts/reproduce_timesnet.sh short_term
bash scripts/reproduce_timesnet.sh imputation all
bash scripts/reproduce_timesnet.sh classification
bash scripts/reproduce_timesnet.sh anomaly all

conda run --no-capture-output -n timesnet python scripts/summarize_timesnet_results.py
```

## 12. 最后怎么判断算“复现成功”

通常按下面的标准判断：

- 长期预测和缺失值填补的平均 `MSE / MAE` 与论文接近。
- M4 的 `SMAPE / MASE / OWA` 接近论文。
- 分类平均准确率接近 `73.6%`。
- 异常检测平均 F1 接近 `85.49%`。
- 单个数据集有轻微浮动是正常的，尤其当 CUDA、PyTorch、小数精度、DataLoader worker 数、随机性略有差异时。

如果你愿意更严格一点，就看 `TIMESNET_REPRODUCTION_RESULTS.md` 里的 `Delta` 列：

- 越接近 `0` 越好。
- 如果某一整个任务普遍偏差较大，优先检查环境、GPU、脚本是否完整跑完、是否误把比例值当成百分数、以及是否把平均值算错了。
