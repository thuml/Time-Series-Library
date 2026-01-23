"""
阶段 3：损失函数测试 (ts_losses)

测试 TimeSeriesAwareLoss 及相关损失函数：
1. 点级损失 (MSE)
2. 趋势损失 (一阶差分)
3. 频域损失 (FFT)
4. 相关性损失 (变量间相关矩阵)
5. 加权总损失
6. CRPS 损失
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


class TestPointLoss:
    """测试 1: 点级 MSE 损失"""

    def test_point_loss(self):
        from utils.ts_losses import TimeSeriesAwareLoss

        loss_fn = TimeSeriesAwareLoss()

        B, T, N = 4, 96, 7
        pred = torch.randn(B, T, N)
        target = torch.randn(B, T, N)

        loss = loss_fn.point_loss(pred, target)

        assert loss >= 0, f"点级损失应 >= 0，实际为 {loss.item()}"
        assert torch.isfinite(loss), "点级损失包含 NaN/Inf"

        # 验证相同输入的损失为 0
        loss_same = loss_fn.point_loss(pred, pred)
        assert loss_same.item() < 1e-7, f"相同输入的损失应为 0，实际为 {loss_same.item()}"

        print(f"✓ test_point_loss 通过 (loss={loss.item():.6f})")


class TestTrendLoss:
    """测试 2: 趋势损失 (一阶差分)"""

    def test_trend_loss(self):
        from utils.ts_losses import TimeSeriesAwareLoss

        loss_fn = TimeSeriesAwareLoss()

        B, T, N = 4, 96, 7
        pred = torch.randn(B, T, N)
        target = torch.randn(B, T, N)

        loss = loss_fn.trend_loss(pred, target)

        assert loss >= 0, f"趋势损失应 >= 0，实际为 {loss.item()}"
        assert torch.isfinite(loss), "趋势损失包含 NaN/Inf"

        # 验证相同趋势的损失为 0
        loss_same = loss_fn.trend_loss(pred, pred)
        assert loss_same.item() < 1e-7, f"相同输入的趋势损失应为 0，实际为 {loss_same.item()}"

        print(f"✓ test_trend_loss 通过 (loss={loss.item():.6f})")


class TestFrequencyLoss:
    """测试 3: 频域损失 (FFT)"""

    def test_frequency_loss(self):
        from utils.ts_losses import TimeSeriesAwareLoss

        loss_fn = TimeSeriesAwareLoss()

        B, T, N = 4, 96, 7
        pred = torch.randn(B, T, N)
        target = torch.randn(B, T, N)

        loss = loss_fn.frequency_loss(pred, target)

        assert loss >= 0, f"频域损失应 >= 0，实际为 {loss.item()}"
        assert torch.isfinite(loss), "频域损失包含 NaN/Inf"

        # 验证相同输入的频域损失为 0
        loss_same = loss_fn.frequency_loss(pred, pred)
        assert loss_same.item() < 1e-7, f"相同输入的频域损失应为 0，实际为 {loss_same.item()}"

        print(f"✓ test_frequency_loss 通过 (loss={loss.item():.6f})")


class TestCorrelationLoss:
    """测试 4: 相关性损失 (变量间相关矩阵)"""

    def test_correlation_loss(self):
        from utils.ts_losses import TimeSeriesAwareLoss

        loss_fn = TimeSeriesAwareLoss()

        B, T, N = 4, 96, 7
        pred = torch.randn(B, T, N)
        target = torch.randn(B, T, N)

        loss = loss_fn.correlation_loss(pred, target)

        assert loss >= 0, f"相关性损失应 >= 0，实际为 {loss.item()}"
        assert torch.isfinite(loss), "相关性损失包含 NaN/Inf"

        # 验证相同输入的相关性损失为 0
        loss_same = loss_fn.correlation_loss(pred, pred)
        assert loss_same.item() < 1e-7, f"相同输入的相关性损失应为 0，实际为 {loss_same.item()}"

        print(f"✓ test_correlation_loss 通过 (loss={loss.item():.6f})")


class TestTotalWeightedLoss:
    """测试 5: 加权总损失"""

    def test_total_weighted_loss(self):
        from utils.ts_losses import TimeSeriesAwareLoss

        # 自定义权重
        loss_fn = TimeSeriesAwareLoss(
            lambda_point=1.0,
            lambda_trend=0.1,
            lambda_freq=0.1,
            lambda_corr=0.05
        )

        B, T, N = 4, 96, 7
        pred = torch.randn(B, T, N)
        target = torch.randn(B, T, N)

        total_loss, loss_dict = loss_fn(pred, target)

        # 验证总损失
        assert total_loss >= 0, f"总损失应 >= 0，实际为 {total_loss.item()}"
        assert torch.isfinite(total_loss), "总损失包含 NaN/Inf"

        # 验证损失字典包含所有项
        expected_keys = {'loss_point', 'loss_trend', 'loss_freq', 'loss_corr', 'loss_total'}
        assert expected_keys == set(loss_dict.keys()), f"损失字典键不匹配: {loss_dict.keys()}"

        # 验证各项损失为正
        for key, value in loss_dict.items():
            assert value >= 0, f"{key} 应 >= 0，实际为 {value}"

        print(f"✓ test_total_weighted_loss 通过")
        print(f"  - point: {loss_dict['loss_point']:.6f}")
        print(f"  - trend: {loss_dict['loss_trend']:.6f}")
        print(f"  - freq: {loss_dict['loss_freq']:.6f}")
        print(f"  - corr: {loss_dict['loss_corr']:.6f}")
        print(f"  - total: {loss_dict['loss_total']:.6f}")


class TestCRPSLoss:
    """测试 6: CRPS 损失"""

    def test_crps_loss(self):
        from utils.ts_losses import CRPSLoss

        crps_fn = CRPSLoss(n_samples=100)

        n_samples, B, T, N = 50, 4, 96, 7
        samples = torch.randn(n_samples, B, T, N)
        target = torch.randn(B, T, N)

        loss = crps_fn(samples, target)

        # CRPS 可以为负值（但在合理范围内）
        assert torch.isfinite(loss), "CRPS 损失包含 NaN/Inf"
        assert loss.item() > -10 and loss.item() < 10, f"CRPS 值异常: {loss.item()}"

        # 验证当样本均值接近目标时，CRPS 应该较小
        target_samples = target.unsqueeze(0).expand(n_samples, -1, -1, -1) + torch.randn_like(samples) * 0.1
        loss_close = crps_fn(target_samples, target)
        assert loss_close.item() < loss.item(), "接近目标的样本 CRPS 应该更小"

        print(f"✓ test_crps_loss 通过 (loss={loss.item():.6f}, close={loss_close.item():.6f})")


if __name__ == '__main__':
    print("=" * 60)
    print("阶段 3：损失函数测试 (ts_losses)")
    print("=" * 60)

    # 运行所有测试
    test_classes = [
        TestPointLoss(),
        TestTrendLoss(),
        TestFrequencyLoss(),
        TestCorrelationLoss(),
        TestTotalWeightedLoss(),
        TestCRPSLoss(),
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
                    import traceback
                    traceback.print_exc()
                    failed += 1

    print("=" * 60)
    print(f"阶段 3 测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        exit(1)
