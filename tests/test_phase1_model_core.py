"""
阶段 1：模型核心测试 (v-prediction)

测试 iTransformerDiffusionDirect 模型的核心功能：
1. 默认参数化为 v
2. v-prediction 目标公式正确
3. x₀ 恢复精度
4. 条件 clamp 逻辑
5. 三种参数化都能恢复 x₀
6. DDPM/DDIM 采样有效
7. 批量采样输出形状
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from argparse import Namespace


def get_test_config():
    """创建测试用配置"""
    return Namespace(
        task_name='diffusion_forecast',
        seq_len=96,
        pred_len=96,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=64,
        d_ff=64,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        factor=1,
        embed='timeF',
        freq='h',
        dropout=0.1,
        activation='gelu',
        diffusion_steps=100,  # 小步数用于测试
        beta_schedule='cosine',
        cond_dim=64,
        unet_channels=[32, 64],
        n_samples=10,
        parameterization='v',
    )


class TestDefaultParameterization:
    """测试 1: 验证默认参数化为 v"""

    def test_default_parameterization_is_v(self):
        from models.iTransformerDiffusionDirect import Model

        configs = get_test_config()
        # 不设置 parameterization，使用默认值
        delattr(configs, 'parameterization')

        model = Model(configs)
        assert model.parameterization == 'v', f"默认参数化应为 'v'，实际为 '{model.parameterization}'"
        print("✓ test_default_parameterization_is_v 通过")


class TestVPredictionFormula:
    """测试 2: 验证 v = √ᾱ·ε - √(1-ᾱ)·x₀"""

    def test_v_prediction_target_formula(self):
        from models.iTransformerDiffusionDirect import Model

        configs = get_test_config()
        configs.parameterization = 'v'
        model = Model(configs)

        B, N, T = 4, 7, 96
        device = 'cpu'

        # 创建测试数据
        x0 = torch.randn(B, N, T, device=device)
        noise = torch.randn(B, N, T, device=device)
        t = torch.randint(0, model.timesteps, (B,), device=device, dtype=torch.long)

        # 计算 v 目标
        sqrt_alpha_cumprod = model.sqrt_alpha_cumprods[t][:, None, None]
        sqrt_one_minus_alpha_cumprod = model.sqrt_one_minus_alpha_cumprods[t][:, None, None]

        # v = √ᾱ·ε - √(1-ᾱ)·x₀
        v_target = sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * x0

        # 验证公式（手动计算）
        alpha_cumprod = model.alpha_cumprods[t][:, None, None]
        v_manual = torch.sqrt(alpha_cumprod) * noise - torch.sqrt(1 - alpha_cumprod) * x0

        assert torch.allclose(v_target, v_manual, atol=1e-6), "v-prediction 公式不正确"
        print("✓ test_v_prediction_target_formula 通过")


class TestX0Recovery:
    """测试 3: 验证从 v 恢复 x₀ 的精度"""

    def test_x0_recovery_from_v(self):
        from models.iTransformerDiffusionDirect import Model

        configs = get_test_config()
        configs.parameterization = 'v'
        model = Model(configs)
        model.eval()

        B, N, T = 4, 7, 96
        device = 'cpu'

        # 创建测试数据
        x0 = torch.randn(B, N, T, device=device)
        noise = torch.randn(B, N, T, device=device)
        t = torch.randint(0, model.timesteps, (B,), device=device, dtype=torch.long)

        # 加噪
        xt, _ = model.add_noise(x0, t, noise)

        # 计算真实的 v
        sqrt_alpha_cumprod = model.sqrt_alpha_cumprods[t][:, None, None]
        sqrt_one_minus_alpha_cumprod = model.sqrt_one_minus_alpha_cumprods[t][:, None, None]
        v_true = sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * x0

        # 从 v 恢复 x₀
        with torch.no_grad():
            x0_recovered = model.predict_x0_from_output(v_true, xt, t)

        # 验证恢复精度
        error = (x0 - x0_recovered).abs().max().item()
        assert error < 1e-5, f"x₀ 恢复误差过大: {error}"
        print(f"✓ test_x0_recovery_from_v 通过 (最大误差: {error:.2e})")


class TestConditionalClamp:
    """测试 4: 验证 clamp 仅在 x0 参数化时执行"""

    def test_conditional_clamp(self):
        from models.iTransformerDiffusionDirect import Model

        # 测试 v-prediction: 不应该 clamp
        configs_v = get_test_config()
        configs_v.parameterization = 'v'
        model_v = Model(configs_v)

        # 测试 x0-prediction: 应该 clamp
        configs_x0 = get_test_config()
        configs_x0.parameterization = 'x0'
        model_x0 = Model(configs_x0)

        # 检查源代码中的 clamp 逻辑
        import inspect
        source = inspect.getsource(model_v.sample_ddpm)

        # 验证 clamp 是有条件的
        assert "if self.parameterization == 'x0'" in source, "clamp 应该是有条件的"
        print("✓ test_conditional_clamp 通过")


class TestAllParameterizationsX0Recovery:
    """测试 5: 验证三种参数化都能正确恢复 x₀"""

    def test_predict_x0_all_parameterizations(self, param_type):
        from models.iTransformerDiffusionDirect import Model

        configs = get_test_config()
        configs.parameterization = param_type
        model = Model(configs)
        model.eval()

        B, N, T = 4, 7, 96
        device = 'cpu'

        # 创建测试数据
        x0 = torch.randn(B, N, T, device=device)
        noise = torch.randn(B, N, T, device=device)
        t = torch.randint(0, model.timesteps, (B,), device=device, dtype=torch.long)

        # 加噪
        xt, _ = model.add_noise(x0, t, noise)

        # 根据参数化类型计算真实输出
        sqrt_alpha_cumprod = model.sqrt_alpha_cumprods[t][:, None, None]
        sqrt_one_minus_alpha_cumprod = model.sqrt_one_minus_alpha_cumprods[t][:, None, None]

        if param_type == 'x0':
            model_output = x0
        elif param_type == 'epsilon':
            model_output = noise
        else:  # v
            model_output = sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * x0

        # 恢复 x₀
        with torch.no_grad():
            x0_recovered = model.predict_x0_from_output(model_output, xt, t)

        # 验证
        error = (x0 - x0_recovered).abs().max().item()
        assert error < 1e-4, f"{param_type} 参数化 x₀ 恢复误差过大: {error}"
        print(f"✓ test_predict_x0_{param_type}_parameterization 通过 (误差: {error:.2e})")


class TestSamplingVPrediction:
    """测试 6: 验证 DDPM/DDIM 采样输出有效"""

    def test_sampling_v_prediction_ddpm(self):
        from models.iTransformerDiffusionDirect import Model

        configs = get_test_config()
        configs.parameterization = 'v'
        configs.diffusion_steps = 20  # 减少步数加速测试
        model = Model(configs)
        model.eval()

        B, N = 2, 7
        device = 'cpu'

        # 模拟编码器特征
        z = torch.randn(B, N, configs.d_model, device=device)

        # DDPM 采样
        with torch.no_grad():
            samples = model.sample_ddpm(z, n_samples=1)

        # 验证输出
        assert samples.shape == (1, B, N, configs.pred_len), f"DDPM 输出形状错误: {samples.shape}"
        assert torch.isfinite(samples).all(), "DDPM 输出包含 NaN/Inf"
        print(f"✓ test_sampling_v_prediction_ddpm 通过 (形状: {samples.shape})")

    def test_sampling_v_prediction_ddim(self):
        from models.iTransformerDiffusionDirect import Model

        configs = get_test_config()
        configs.parameterization = 'v'
        configs.diffusion_steps = 20
        model = Model(configs)
        model.eval()

        B, N = 2, 7
        device = 'cpu'

        z = torch.randn(B, N, configs.d_model, device=device)

        # DDIM 采样
        with torch.no_grad():
            samples = model.sample_ddim(z, n_samples=1, ddim_steps=5)

        assert samples.shape == (1, B, N, configs.pred_len), f"DDIM 输出形状错误: {samples.shape}"
        assert torch.isfinite(samples).all(), "DDIM 输出包含 NaN/Inf"
        print(f"✓ test_sampling_v_prediction_ddim 通过 (形状: {samples.shape})")


class TestBatchSampling:
    """测试 7: 验证批量采样输出形状正确"""

    def test_batch_sampling(self):
        from models.iTransformerDiffusionDirect import Model

        configs = get_test_config()
        configs.parameterization = 'v'
        configs.diffusion_steps = 20
        model = Model(configs)
        model.eval()

        B, N, n_samples = 2, 7, 5
        device = 'cpu'

        z = torch.randn(B, N, configs.d_model, device=device)

        # 批量 DDPM 采样
        with torch.no_grad():
            samples_ddpm = model.sample_ddpm_x0_batch(z, n_samples=n_samples)

        expected_shape = (n_samples, B, N, configs.pred_len)
        assert samples_ddpm.shape == expected_shape, f"批量 DDPM 形状错误: {samples_ddpm.shape} vs {expected_shape}"
        assert torch.isfinite(samples_ddpm).all(), "批量 DDPM 输出包含 NaN/Inf"

        # 批量 DDIM 采样
        with torch.no_grad():
            samples_ddim = model.sample_ddim_x0_batch(z, n_samples=n_samples, ddim_steps=5)

        assert samples_ddim.shape == expected_shape, f"批量 DDIM 形状错误: {samples_ddim.shape} vs {expected_shape}"
        assert torch.isfinite(samples_ddim).all(), "批量 DDIM 输出包含 NaN/Inf"

        print(f"✓ test_batch_sampling 通过 (形状: {expected_shape})")


if __name__ == '__main__':
    print("=" * 60)
    print("阶段 1：模型核心测试 (v-prediction)")
    print("=" * 60)

    # 运行所有测试
    test_classes = [
        TestDefaultParameterization(),
        TestVPredictionFormula(),
        TestX0Recovery(),
        TestConditionalClamp(),
        TestSamplingVPrediction(),
        TestBatchSampling(),
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                try:
                    getattr(test_class, method_name)()
                    passed += 1
                except Exception as e:
                    print(f"✗ {method_name} 失败: {e}")
                    failed += 1

    # 参数化测试
    test_all_param = TestAllParameterizationsX0Recovery()
    for param_type in ['x0', 'epsilon', 'v']:
        try:
            test_all_param.test_predict_x0_all_parameterizations(param_type)
            passed += 1
        except Exception as e:
            print(f"✗ test_predict_x0_{param_type} 失败: {e}")
            failed += 1

    print("=" * 60)
    print(f"阶段 1 测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        exit(1)
