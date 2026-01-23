"""
FR2 模块验证脚本（不依赖 pytest）

快速验证 FR2 模块的基本功能。
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.Diffusion_layers import FrequencyAwareResidual


def test_fr2_basic():
    """基础功能测试"""
    print("=" * 60)
    print("FR2 (Frequency-aware Residual) 模块验证")
    print("=" * 60)

    # 参数
    B, C, T = 4, 128, 96
    N, d_model = 7, 128
    n_freqs = 10

    print(f"\n配置:")
    print(f"  Batch size: {B}")
    print(f"  Channels: {C}")
    print(f"  Time steps: {T}")
    print(f"  Variates: {N}")
    print(f"  d_model: {d_model}")
    print(f"  n_freqs: {n_freqs}")

    # 初始化
    print("\n[1] 初始化 FR2 模块...")
    fr2 = FrequencyAwareResidual(
        channels=C,
        d_model=d_model,
        n_freqs=n_freqs
    )
    print("    ✓ 初始化成功")

    # 前向传播
    print("\n[2] 前向传播测试...")
    x = torch.randn(B, C, T)
    z = torch.randn(B, N, d_model)
    print(f"    输入 x: {x.shape}")
    print(f"    输入 z: {z.shape}")

    out = fr2(x, z)
    print(f"    输出: {out.shape}")

    # 验证形状
    assert out.shape == (B, C, T), f"形状错误: 期望 {(B, C, T)}, 实际 {out.shape}"
    print("    ✓ 输出形状正确")

    # 验证数值稳定性
    print("\n[3] 数值稳定性测试...")
    assert not torch.isnan(out).any(), "输出包含 NaN"
    assert not torch.isinf(out).any(), "输出包含 Inf"
    print("    ✓ 无 NaN/Inf")

    # 验证频域调制
    print("\n[4] 频域调制效果测试...")
    t = torch.linspace(0, 2 * torch.pi, T)
    x_sine = torch.sin(5 * t).unsqueeze(0).unsqueeze(0).expand(B, C, -1)
    out_sine = fr2(x_sine, z)

    # 输出应该与输入不同
    diff = (out_sine - x_sine).abs().mean().item()
    print(f"    平均差异: {diff:.6f}")
    assert diff > 1e-6, "输出应该与输入有明显差异"
    print("    ✓ 频域调制生效")

    # 验证梯度流
    print("\n[5] 梯度流测试...")
    x_grad = torch.randn(B, C, T, requires_grad=True)
    z_grad = torch.randn(B, N, d_model, requires_grad=True)

    out_grad = fr2(x_grad, z_grad)
    loss = out_grad.mean()
    loss.backward()

    assert x_grad.grad is not None, "输入梯度不存在"
    assert z_grad.grad is not None, "条件梯度不存在"
    assert torch.isfinite(x_grad.grad).all(), "输入梯度包含 NaN/Inf"
    assert torch.isfinite(z_grad.grad).all(), "条件梯度包含 NaN/Inf"
    print(f"    x 梯度范数: {x_grad.grad.norm().item():.6f}")
    print(f"    z 梯度范数: {z_grad.grad.norm().item():.6f}")
    print("    ✓ 梯度流正常")

    # 验证不同频率分辨率
    print("\n[6] 不同频率分辨率测试...")
    for nf in [5, 10, 20]:
        fr2_temp = FrequencyAwareResidual(C, d_model, nf)
        out_temp = fr2_temp(x, z)
        assert out_temp.shape == (B, C, T), f"n_freqs={nf} 时形状错误"
        assert torch.isfinite(out_temp).all(), f"n_freqs={nf} 时数值不稳定"
        print(f"    ✓ n_freqs={nf} 工作正常")

    print("\n" + "=" * 60)
    print("所有测试通过！FR2 模块工作正常 ✓")
    print("=" * 60)


if __name__ == '__main__':
    test_fr2_basic()
