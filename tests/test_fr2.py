"""
单元测试: FR2 (Frequency-aware Residual) 模块

测试内容:
1. 模块初始化和前向传播
2. 输出形状正确性
3. 数值稳定性（无 NaN/Inf）
4. 频域调制效果
5. 门控机制工作正常
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers.Diffusion_layers import FrequencyAwareResidual


class TestFR2:
    """FR2 模块测试套件"""

    def setup_method(self):
        """每个测试前的设置"""
        self.B = 4  # batch size
        self.C = 128  # channels (from UNet bottleneck)
        self.T = 96  # time steps (pred_len)
        self.N = 7  # number of variates
        self.d_model = 128  # backbone feature dimension
        self.n_freqs = 10  # frequency resolution

    def test_initialization(self):
        """测试模块初始化"""
        fr2 = FrequencyAwareResidual(
            channels=self.C,
            d_model=self.d_model,
            n_freqs=self.n_freqs
        )

        assert fr2.n_freqs == self.n_freqs
        assert hasattr(fr2, 'freq_proj')
        assert hasattr(fr2, 'residual_gate')

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        fr2 = FrequencyAwareResidual(
            channels=self.C,
            d_model=self.d_model,
            n_freqs=self.n_freqs
        )

        # 输入
        x = torch.randn(self.B, self.C, self.T)  # 扩散特征
        z = torch.randn(self.B, self.N, self.d_model)  # backbone 特征

        # 前向传播
        out = fr2(x, z)

        # 验证输出形状
        assert out.shape == (self.B, self.C, self.T), \
            f"Expected shape {(self.B, self.C, self.T)}, got {out.shape}"

    def test_numerical_stability(self):
        """测试数值稳定性（无 NaN/Inf）"""
        fr2 = FrequencyAwareResidual(
            channels=self.C,
            d_model=self.d_model,
            n_freqs=self.n_freqs
        )

        # 输入
        x = torch.randn(self.B, self.C, self.T)
        z = torch.randn(self.B, self.N, self.d_model)

        # 前向传播
        out = fr2(x, z)

        # 验证无 NaN/Inf
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_frequency_modulation(self):
        """测试频域调制效果"""
        fr2 = FrequencyAwareResidual(
            channels=self.C,
            d_model=self.d_model,
            n_freqs=self.n_freqs
        )

        # 创建纯正弦波输入（已知频域特性）
        t = torch.linspace(0, 2 * torch.pi, self.T)
        x = torch.sin(5 * t).unsqueeze(0).unsqueeze(0).expand(self.B, self.C, -1)
        z = torch.randn(self.B, self.N, self.d_model)

        # 前向传播
        out = fr2(x, z)

        # 验证输出不等于输入（说明发生了调制）
        assert not torch.allclose(out, x, atol=1e-6), \
            "Output should be different from input (frequency modulation should occur)"

        # 验证输出形状不变
        assert out.shape == x.shape

    def test_gate_mechanism(self):
        """测试门控机制工作正常"""
        fr2 = FrequencyAwareResidual(
            channels=self.C,
            d_model=self.d_model,
            n_freqs=self.n_freqs
        )

        # 零输入 backbone 特征
        x = torch.randn(self.B, self.C, self.T)
        z = torch.zeros(self.B, self.N, self.d_model)

        # 前向传播
        out = fr2(x, z)

        # 当 z=0 时，频域参数应该接近0，门控应该接近 0.5（sigmoid(0)）
        # 因此输出应该接近原始输入（但不完全相等）
        # 这里只验证输出是有限的且形状正确
        assert torch.isfinite(out).all(), "Output should be finite"
        assert out.shape == x.shape

    def test_different_n_freqs(self):
        """测试不同的频率分辨率"""
        for n_freqs in [5, 10, 20]:
            fr2 = FrequencyAwareResidual(
                channels=self.C,
                d_model=self.d_model,
                n_freqs=n_freqs
            )

            x = torch.randn(self.B, self.C, self.T)
            z = torch.randn(self.B, self.N, self.d_model)

            out = fr2(x, z)

            assert out.shape == (self.B, self.C, self.T)
            assert torch.isfinite(out).all()

    def test_batch_independence(self):
        """测试 batch 样本之间的独立性"""
        fr2 = FrequencyAwareResidual(
            channels=self.C,
            d_model=self.d_model,
            n_freqs=self.n_freqs
        )

        # 创建两个不同的 batch
        x1 = torch.randn(1, self.C, self.T)
        x2 = torch.randn(1, self.C, self.T)
        x_concat = torch.cat([x1, x2], dim=0)

        z1 = torch.randn(1, self.N, self.d_model)
        z2 = torch.randn(1, self.N, self.d_model)
        z_concat = torch.cat([z1, z2], dim=0)

        # 分别处理
        out1 = fr2(x1, z1)
        out2 = fr2(x2, z2)

        # 一起处理
        out_concat = fr2(x_concat, z_concat)

        # 验证结果一致（允许小的数值误差）
        assert torch.allclose(out1, out_concat[0:1], atol=1e-5)
        assert torch.allclose(out2, out_concat[1:2], atol=1e-5)

    def test_gradient_flow(self):
        """测试梯度流正常"""
        fr2 = FrequencyAwareResidual(
            channels=self.C,
            d_model=self.d_model,
            n_freqs=self.n_freqs
        )

        x = torch.randn(self.B, self.C, self.T, requires_grad=True)
        z = torch.randn(self.B, self.N, self.d_model, requires_grad=True)

        # 前向传播
        out = fr2(x, z)

        # 计算损失并反向传播
        loss = out.mean()
        loss.backward()

        # 验证梯度存在且有限
        assert x.grad is not None, "Input gradient should exist"
        assert z.grad is not None, "Condition gradient should exist"
        assert torch.isfinite(x.grad).all(), "Input gradient should be finite"
        assert torch.isfinite(z.grad).all(), "Condition gradient should be finite"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
