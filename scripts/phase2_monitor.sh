#!/bin/bash

# Phase 2 训练监控脚本
# 实时查看所有训练任务的进度

echo "=========================================="
echo "Phase 2 训练监控"
echo "=========================================="
echo ""

# 检查进程状态
echo "=== 进程状态 ==="
RUNNING=$(ps aux | grep "[p]ython.*run.py" | wc -l)
echo "运行中的训练任务: $RUNNING"
echo ""

# 检查日志文件大小
echo "=== 日志文件大小 ==="
ls -lh logs/phase2/*.log 2>/dev/null || echo "日志文件不存在"
echo ""

# 显示各任务的最新进度
echo "=== 基线训练 (iTransformerDiffusionDirect) ==="
if [ -f logs/phase2/baseline.log ]; then
    grep -E "(Epoch:|Best)" logs/phase2/baseline.log | tail -5
else
    echo "日志文件不存在"
fi
echo ""

echo "=== iTransformer 对比训练 ==="
if [ -f logs/phase2/iTransformer.log ]; then
    grep -E "(Epoch:|Best)" logs/phase2/iTransformer.log | tail -5
else
    echo "日志文件不存在"
fi
echo ""

echo "=== PatchTST 对比训练 ==="
if [ -f logs/phase2/PatchTST.log ]; then
    grep -E "(Epoch:|Best)" logs/phase2/PatchTST.log | tail -5
else
    echo "日志文件不存在"
fi
echo ""

echo "=== TimesNet 对比训练 ==="
if [ -f logs/phase2/TimesNet.log ]; then
    grep -E "(Epoch:|Best)" logs/phase2/TimesNet.log | tail -5
else
    echo "日志文件不存在"
fi
echo ""

# GPU 使用情况
echo "=== GPU 使用情况 ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi 不可用"
echo ""

# 训练完成检查
echo "=== 训练完成检查 ==="
for model in baseline iTransformer PatchTST TimesNet; do
    if [ -f "logs/phase2/${model}.log" ]; then
        if grep -q "Training completed" logs/phase2/${model}.log 2>/dev/null; then
            echo "✓ ${model} - 训练完成"
        else
            echo "⏳ ${model} - 训练中..."
        fi
    fi
done
echo ""

echo "=========================================="
echo "监控命令:"
echo "  持续监控: watch -n 30 bash scripts/phase2_monitor.sh"
echo "  查看实时日志: tail -f logs/phase2/baseline.log"
echo "  查看所有日志: tail -f logs/phase2/*.log"
echo "=========================================="
