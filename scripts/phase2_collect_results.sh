#!/bin/bash

# Phase 2 结果收集脚本
# 训练完成后运行此脚本自动提取所有模型的指标

echo "=========================================="
echo "Phase 2 结果收集"
echo "=========================================="
echo ""

# 检查训练是否完成
echo "=== 训练状态检查 ==="
ALL_DONE=true
for model in baseline iTransformer PatchTST TimesNet; do
    if [ -f "logs/phase2/${model}.log" ]; then
        if grep -q "Training completed\|>>>>>>>testing" logs/phase2/${model}.log 2>/dev/null; then
            echo "✓ ${model} - 已完成"
        else
            echo "⏳ ${model} - 仍在训练中"
            ALL_DONE=false
        fi
    else
        echo "✗ ${model} - 日志文件不存在"
        ALL_DONE=false
    fi
done
echo ""

if [ "$ALL_DONE" = false ]; then
    echo "警告: 部分训练尚未完成"
    echo "建议等待所有训练完成后再收集结果"
    echo ""
fi

# 查找结果文件
echo "=== 结果文件 ==="
echo ""

echo "基线模型 (iTransformerDiffusionDirect):"
BASELINE_DIR=$(find checkpoints/ -type d -name "*diffusion_forecast_ETTh1_96_96_baseline*" | head -1)
if [ -n "$BASELINE_DIR" ]; then
    echo "  目录: $BASELINE_DIR"
    if [ -f "$BASELINE_DIR/result.txt" ]; then
        echo "  结果:"
        cat "$BASELINE_DIR/result.txt" | head -20
    else
        echo "  结果文件不存在"
    fi
else
    echo "  Checkpoint 目录不存在"
fi
echo ""

echo "iTransformer:"
ITRANS_DIR=$(find checkpoints/ -type d -name "*long_term_forecast_ETTh1_96_96_baseline_iTransformer*" | head -1)
if [ -n "$ITRANS_DIR" ]; then
    echo "  目录: $ITRANS_DIR"
    if [ -f "$ITRANS_DIR/result.txt" ]; then
        echo "  结果:"
        cat "$ITRANS_DIR/result.txt" | head -10
    else
        echo "  结果文件不存在"
    fi
else
    echo "  Checkpoint 目录不存在"
fi
echo ""

echo "PatchTST:"
PATCH_DIR=$(find checkpoints/ -type d -name "*long_term_forecast_ETTh1_96_96_baseline_PatchTST*" | head -1)
if [ -n "$PATCH_DIR" ]; then
    echo "  目录: $PATCH_DIR"
    if [ -f "$PATCH_DIR/result.txt" ]; then
        echo "  结果:"
        cat "$PATCH_DIR/result.txt" | head -10
    else
        echo "  结果文件不存在"
    fi
else
    echo "  Checkpoint 目录不存在"
fi
echo ""

echo "TimesNet:"
TIMES_DIR=$(find checkpoints/ -type d -name "*long_term_forecast_ETTh1_96_96_baseline_TimesNet*" | head -1)
if [ -n "$TIMES_DIR" ]; then
    echo "  目录: $TIMES_DIR"
    if [ -f "$TIMES_DIR/result.txt" ]; then
        echo "  结果:"
        cat "$TIMES_DIR/result.txt" | head -10
    else
        echo "  结果文件不存在"
    fi
else
    echo "  Checkpoint 目录不存在"
fi
echo ""

# 提取关键指标
echo "=========================================="
echo "快速对比表格"
echo "=========================================="
echo ""
printf "%-30s | %-10s | %-10s | %-10s\n" "模型" "MSE" "MAE" "备注"
echo "-----------------------------------------------------------"

# 提取 iTransformer
if [ -f "$ITRANS_DIR/result.txt" ]; then
    MSE=$(grep "mse:" "$ITRANS_DIR/result.txt" | awk '{print $2}' | head -1)
    MAE=$(grep "mae:" "$ITRANS_DIR/result.txt" | awk '{print $2}' | head -1)
    printf "%-30s | %-10s | %-10s | %-10s\n" "iTransformer" "$MSE" "$MAE" "Backbone"
fi

# 提取 PatchTST
if [ -f "$PATCH_DIR/result.txt" ]; then
    MSE=$(grep "mse:" "$PATCH_DIR/result.txt" | awk '{print $2}' | head -1)
    MAE=$(grep "mae:" "$PATCH_DIR/result.txt" | awk '{print $2}' | head -1)
    printf "%-30s | %-10s | %-10s | %-10s\n" "PatchTST" "$MSE" "$MAE" "Patch-based"
fi

# 提取 TimesNet
if [ -f "$TIMES_DIR/result.txt" ]; then
    MSE=$(grep "mse:" "$TIMES_DIR/result.txt" | awk '{print $2}' | head -1)
    MAE=$(grep "mae:" "$TIMES_DIR/result.txt" | awk '{print $2}' | head -1)
    printf "%-30s | %-10s | %-10s | %-10s\n" "TimesNet" "$MSE" "$MAE" "Multi-period"
fi

# 提取基线
if [ -f "$BASELINE_DIR/result.txt" ]; then
    MSE=$(grep "mse:" "$BASELINE_DIR/result.txt" | awk '{print $2}' | head -1)
    MAE=$(grep "mae:" "$BASELINE_DIR/result.txt" | awk '{print $2}' | head -1)
    CRPS=$(grep "crps:" "$BASELINE_DIR/result.txt" | awk '{print $2}' | head -1)
    printf "%-30s | %-10s | %-10s | %-10s\n" "Baseline (ours)" "$MSE" "$MAE" "Probabilistic"
    if [ -n "$CRPS" ]; then
        echo ""
        echo "基线模型额外指标:"
        echo "  CRPS: $CRPS"
        grep "calibration\|coverage" "$BASELINE_DIR/result.txt" 2>/dev/null | head -5
    fi
fi

echo ""
echo "=========================================="
echo "详细结果文件位置:"
echo "  基线: $BASELINE_DIR/result.txt"
echo "  iTransformer: $ITRANS_DIR/result.txt"
echo "  PatchTST: $PATCH_DIR/result.txt"
echo "  TimesNet: $TIMES_DIR/result.txt"
echo "=========================================="
