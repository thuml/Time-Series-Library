#!/bin/bash

# Phase 2: 并行启动所有训练任务
# 包括: 基线模型 + 3 个对比模型
#
# 使用方法:
#   bash scripts/phase2_launch_all.sh
#
# 预计总训练时间: 4-6 小时（并行执行）
# 单个模型时间: 2-3 小时（30 epoch）

echo "=========================================="
echo "Phase 2: 启动所有训练任务"
echo "=========================================="
echo ""

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tslib

# 创建日志目录
mkdir -p logs/phase2

# 任务 1: iTransformerDiffusionDirect 基线训练
echo "[1/4] 启动 iTransformerDiffusionDirect 基线训练..."
nohup bash scripts/diffusion_forecast/ETT_script/iTransformerDiffusionDirect_ETTh1_baseline.sh \
  > logs/phase2/baseline.log 2>&1 &
PID_BASELINE=$!
echo "      PID: $PID_BASELINE"
echo "      日志: logs/phase2/baseline.log"

sleep 2

# 任务 2: iTransformer 对比训练
echo "[2/4] 启动 iTransformer 对比训练..."
nohup bash scripts/long_term_forecast/ETT_script/iTransformer_ETTh1.sh \
  > logs/phase2/iTransformer.log 2>&1 &
PID_ITRANS=$!
echo "      PID: $PID_ITRANS"
echo "      日志: logs/phase2/iTransformer.log"

sleep 2

# 任务 3: PatchTST 对比训练
echo "[3/4] 启动 PatchTST 对比训练..."
nohup bash scripts/long_term_forecast/ETT_script/PatchTST_ETTh1_baseline.sh \
  > logs/phase2/PatchTST.log 2>&1 &
PID_PATCH=$!
echo "      PID: $PID_PATCH"
echo "      日志: logs/phase2/PatchTST.log"

sleep 2

# 任务 4: TimesNet 对比训练
echo "[4/4] 启动 TimesNet 对比训练..."
nohup bash scripts/long_term_forecast/ETT_script/TimesNet_ETTh1_baseline.sh \
  > logs/phase2/TimesNet.log 2>&1 &
PID_TIMES=$!
echo "      PID: $PID_TIMES"
echo "      日志: logs/phase2/TimesNet.log"

echo ""
echo "=========================================="
echo "所有训练任务已启动！"
echo "=========================================="
echo ""
echo "进程 ID:"
echo "  Baseline (iTransformerDiffusionDirect): $PID_BASELINE"
echo "  iTransformer: $PID_ITRANS"
echo "  PatchTST: $PID_PATCH"
echo "  TimesNet: $PID_TIMES"
echo ""
echo "监控命令:"
echo "  查看所有日志: tail -f logs/phase2/*.log"
echo "  查看基线日志: tail -f logs/phase2/baseline.log"
echo "  查看进程状态: ps aux | grep python"
echo "  杀死所有训练: kill $PID_BASELINE $PID_ITRANS $PID_PATCH $PID_TIMES"
echo ""
echo "预计完成时间: $(date -d '+6 hours' '+%Y-%m-%d %H:%M:%S')"
echo ""
