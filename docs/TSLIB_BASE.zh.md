# TSLib 分支基础

- `main`：TSLib 原版，保留上游完整历史。
- `castle_1023`：基于 main 导入现有 PatchTST / CEEMDAN / TimeXer 项目。
- `origin`：https://github.com/Castle47/Time-Series-Library.git
- `upstream`：https://github.com/thuml/Time-Series-Library.git

当前实验继续使用 `run_longExp.py` 和既有 scripts。重名模块保留当前项目版本，TSLib 的其他模型和脚本保留在仓库中。此次完成 Git 基础迁移，尚未完成与 TSLib `run.py` 多任务接口的全面适配，不能据此认定所有上游模型均可直接运行。

数据、日志、模型权重和实验输出保留本地，不提交到 GitHub。现有项目的 Apache 许可证保存在 `LICENSE.PatchTST`；TSLib 的 MIT 许可证保存在 `LICENSE`。来源细节见 `UPSTREAM.json`。

验证：`.venv/bin/python -m pytest -q`，64 passed（2026-10-23）。
