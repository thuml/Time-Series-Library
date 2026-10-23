# TimeXer / Traffic 验证记录

日期：2026-10-22。运行环境：项目 `.venv`，CPU。

## 官方实现对照

源代码取自 `thuml/TimeXer`，原始文件 SHA-256 记录在 `UPSTREAM.json`。本地复用现有注意力层，官方调用中未使用的 `tau/delta` 参数已移除；独立引入官方 inverted embedding。固定相同权重与输入，比较 `M`、`MS`、`S` × `use_norm=0/1` 六种情况，所有状态字典严格匹配，预测最大绝对误差均为 **0**。对照使用 `seq_len=35`、`patch_len=8`，也覆盖不足一个 patch 的尾段行为。

官方代码的 `x_enc /= stdev` 在输入本身需要梯度时会触发原地修改错误。本地采用数值等价的 `x_enc = x_enc / stdev`，六种情况均通过输入梯度与模型参数梯度检查。

## 自动检查

运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

全部 **57 项通过**，包含新增的 18 项检查：三种预测模式及标准化开关、全局 token 和外生嵌入的梯度、严格保存重载、目标与外生通道分离、对历史外生变量的响应、未来解码输入隔离、仅全局 token 参与交叉注意力、非整除 patch、注意力输出接口、非法配置、Traffic 各分区边界、训练段标准化、目标列顺序、验证和测试尾批保留。

## 真实 Traffic 运行

文件：`data/traffic/traffic.csv`。17,544 行、862 个数值通道、唯一时间戳，全部数值有限。最后一列 `OT` 是最后一个传感器；源文件和下载方式见 `data/traffic/SOURCE.json`。在 `seq_len=96`、`pred_len=96` 下，训练、验证、测试分别有 12,089、1,661、3,413 个窗口。

两种模式都执行了：2 个训练批次 → 2 个验证批次选模 → 保存和重载最佳权重 → 2 个测试批次 → 单个未来预测窗口。批大小 2、模型宽度 16，`M` 额外启用 `--output_attention` 检查训练入口的 tuple 处理。运行日志分别为 `logs/smoke_traffic_timexer.log` 和 `logs/smoke_traffic_timexer_M.log`，预测与指标保存到现有 `results/` 目录，均标记 `limited_run=true`。

这些运行只确认实现和端到端流程可用，未执行完整训练，也不构成论文性能复现。训练脚本中的参数是本项目可运行的起点。
