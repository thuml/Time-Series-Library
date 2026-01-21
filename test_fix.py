#!/usr/bin/env python3
"""
测试 iTransformerDiffusionDirect 模型修复
"""

import torch
import argparse
import sys
import os

# 添加项目路径
sys.path.append("/home/cloud_lin/projects/Time-Series-Library")


def test_model_import():
    """测试模型导入"""
    print("测试 1: 模型导入...")
    try:
        from models.iTransformerDiffusionDirect import (
            Model,
            iTransformerDiffusionDirect,
        )

        print("✓ 模型导入成功")
        return True
    except Exception as e:
        print(f"✗ 模型导入失败: {e}")
        return False


def test_model_registry():
    """测试模型注册"""
    print("测试 2: 模型注册...")
    try:
        # 直接检查 exp_basic.py 文件内容
        with open(
            "/home/cloud_lin/projects/Time-Series-Library/exp/exp_basic.py", "r"
        ) as f:
            content = f.read()

        if '"iTransformerDiffusionDirect": iTransformerDiffusionDirect' in content:
            print("✓ 模型已注册到 model_dict")
            return True
        else:
            print("✗ 模型未注册到 model_dict")
            return False
    except Exception as e:
        print(f"✗ 模型注册测试失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 模型注册测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ 模型注册测试失败: {e}")
        return False


def test_model_instantiation():
    """测试模型实例化"""
    print("测试 3: 模型实例化...")
    try:
        from models.iTransformerDiffusionDirect import Model
        import argparse

        # 创建模拟配置
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
        configs.n_samples = 100
        configs.parameterization = "x0"  # 测试 x0 参数化

        model = Model(configs)
        print("✓ 模型实例化成功")
        return True
    except Exception as e:
        print(f"✗ 模型实例化失败: {e}")
        return False


def test_different_parameterizations():
    """测试不同参数化类型"""
    print("测试 4: 不同参数化类型...")
    try:
        from models.iTransformerDiffusionDirect import Model
        import argparse

        base_configs = argparse.Namespace()
        base_configs.task_name = "diffusion_forecast"
        base_configs.seq_len = 96
        base_configs.pred_len = 96
        base_configs.enc_in = 7
        base_configs.dec_in = 7
        base_configs.c_out = 7
        base_configs.d_model = 128
        base_configs.d_ff = 128
        base_configs.e_layers = 2
        base_configs.n_heads = 8
        base_configs.embed = "timeF"
        base_configs.freq = "h"
        base_configs.dropout = 0.1
        base_configs.activation = "gelu"
        base_configs.factor = 1
        base_configs.diffusion_steps = 1000
        base_configs.beta_schedule = "cosine"
        base_configs.cond_dim = 256
        base_configs.unet_channels = [64, 128, 256, 512]
        base_configs.n_samples = 100

        for param_type in ["x0", "epsilon", "v"]:
            configs = base_configs
            configs.parameterization = param_type
            model = Model(configs)
            if model.parameterization != param_type:
                print(f"✗ 参数化 {param_type} 设置失败")
                return False
            print(f"✓ 参数化 {param_type} 实例化成功")

        return True
    except Exception as e:
        print(f"✗ 参数化测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("开始测试 iTransformerDiffusionDirect 修复...")
    print("=" * 50)

    tests = [
        test_model_import,
        test_model_registry,
        test_model_instantiation,
        test_different_parameterizations,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！模型修复成功。")
        return True
    else:
        print("❌ 部分测试失败，需要进一步修复。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
