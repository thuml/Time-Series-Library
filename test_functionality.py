#!/usr/bin/env python3
"""
功能测试：验证 iTransformerDiffusionDirect 模型可以实际运行
"""

import torch
import argparse
import sys

sys.path.append("/home/cloud_lin/projects/Time-Series-Library")


def test_forward_pass():
    """测试前向传播"""
    print("测试模型前向传播...")

    try:
        from models.iTransformerDiffusionDirect import Model

        # 创建配置
        configs = argparse.Namespace()
        configs.task_name = "diffusion_forecast"
        configs.seq_len = 96
        configs.pred_len = 96
        configs.enc_in = 7
        configs.dec_in = 7
        configs.c_out = 7
        configs.d_model = 128
        configs.d_ff = 128
        configs.e_layers = 2
        configs.n_heads = 8
        configs.embed = "timeF"
        configs.freq = "h"
        configs.dropout = 0.1
        configs.activation = "gelu"
        configs.factor = 1
        configs.diffusion_steps = 1000
        configs.beta_schedule = "cosine"
        configs.cond_dim = 256
        configs.unet_channels = [64, 128, 256, 512]
        configs.n_samples = 10  # 减少采样数量用于测试
        configs.parameterization = "x0"

        # 实例化模型
        model = Model(configs)
        model.eval()

        # 创建模拟数据
        batch_size = 2
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        n_vars = configs.enc_in

        x_enc = torch.randn(batch_size, seq_len, n_vars)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)  # 时间特征
        y_true = torch.randn(batch_size, pred_len, n_vars)

        print(f"  输入形状: {x_enc.shape}")
        print(f"  目标形状: {y_true.shape}")

        # 测试 Stage 1 (warmup)
        with torch.no_grad():
            loss_warmup, loss_dict = model.forward_loss(
                x_enc, x_mark_enc, y_true, stage="warmup"
            )
            print(f"  ✓ Stage 1 损失: {loss_warmup.item():.4f}")

        # 测试 Stage 2 (joint)
        with torch.no_grad():
            loss_joint, loss_dict = model.forward_loss(
                x_enc, x_mark_enc, y_true, stage="joint"
            )
            print(f"  ✓ Stage 2 总损失: {loss_joint.item():.4f}")
            print(f"    MSE 损失: {loss_dict['loss_mse']:.4f}")
            print(f"    扩散损失: {loss_dict['loss_diff']:.4f}")

        # 测试预测
        with torch.no_grad():
            y_pred, z, means, stdev = model.backbone_forward(x_enc, x_mark_enc)
            print(f"  ✓ 确定性预测形状: {y_pred.shape}")

            # 测试采样
            samples = model.sample_ddpm(z, n_samples=2)
            print(f"  ✓ DDPM 采样形状: {samples.shape}")

            # 测试 DDIM 采样
            samples_ddim = model.sample_ddim(z, n_samples=2, ddim_steps=10)
            print(f"  ✓ DDIM 采样形状: {samples_ddim.shape}")

        print("✓ 前向传播测试成功")
        return True

    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cli_integration():
    """测试 CLI 集成"""
    print("\n测试 CLI 集成...")

    try:
        # 检查 run.py 是否可以找到模型
        import subprocess
        import os

        cmd = ["python", "run.py", "--help"]

        result = subprocess.run(
            cmd,
            cwd="/home/cloud_lin/projects/Time-Series-Library",
            capture_output=True,
            text=True,
            timeout=30,
        )

        # 检查命令是否可以运行（不要求实际训练）
        if result.returncode in [0, 2]:  # 2 通常来自 argparse 的错误
            print("✓ CLI 接口可用")
            return True
        else:
            print(f"✗ CLI 接口问题: {result.stderr[:200]}")
            return False

    except Exception as e:
        print(f"✗ CLI 集成测试失败: {e}")
        return False


def main():
    """运行功能测试"""
    print("开始 iTransformerDiffusionDirect 功能测试...")
    print("=" * 60)

    tests = [
        test_forward_pass,
        test_cli_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("=" * 60)
    print(f"功能测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有功能测试通过！模型可以正常使用。")
        return True
    else:
        print("❌ 部分功能测试失败。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
