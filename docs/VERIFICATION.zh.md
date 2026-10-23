# 实现验证记录（2026-10-21）

本次完成结构实现与可运行性验证，未完成用于模型优劣比较的完整训练。

- 26 项自动检查通过：真实 PyEMD 一致性、噪声复用一致性、分量重构、常量/单调/短序列、固定余项的梯度约定、每层两次分解、主要预测路径梯度、batch 与通道独立性、奇数 patch 数周期聚合、9 种消融组合及权重重载。
- 关闭分解并仅保留自注意力后，复制相同权重，输出与官方 PatchTST 的实际默认 BatchNorm 结构一致。
- 官方 PatchTST 基线和完整融合均在 ETTh1 OT 的 96→24 小规模训练中跑通 Adam、OneCycleLR、验证选择、最优权重保存/重载和预测保存。每个模型仅 2 个 epoch、每 epoch 2 个 batch，验证/测试各 1 个 batch，属于功能检查。
- 融合模型保存的预测与独立重载 checkpoint 后重新预测一致，MSE/MAE 可从保存的预测和标签重算；数据 SHA-256 与来源记录一致。机器记录见 [verification.json](verification.json)。
- 多变量检查使用 ETTh1 7 个变量，B=4、L=96、H=96、D=16、两层、8 trials、最多 3 个 IMF：输出 `[4,96,7]` 有限，成功反向传播，4 个分解点实际执行共 1,792 次 CEEMDAN。当前 Mac CPU 上此次前向约 **10.35 秒**，反向约 **0.019 秒**。这只是单批耗时，不能作为完整训练时间或精度结论。见 [multivariate_check.json](multivariate_check.json)。

独立环境已经安装在项目 `.venv`；精确版本见 `requirements.lock.txt`。检查源码指纹见 [verified_sources.sha256.json](verified_sources.sha256.json)。前一个项目的结果不用于本版结构的效果声明；需要完整训练、多个随机种子和对应消融后才能判断是否超过 PatchTST 或 Autoformer。
