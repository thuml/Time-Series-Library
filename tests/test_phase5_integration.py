"""
阶段 5：集成测试

端到端集成测试：
1. 完整训练步骤 (warmup + joint)
2. 概率预测输出形状和有限性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from argparse import Namespace


def get_integration_test_config():
    """创建集成测试配置"""
    return Namespace(
        task_name='diffusion_forecast',
        seq_len=96,
        label_len=48,
        pred_len=96,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=32,  # 小模型加速测试
        d_ff=32,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        factor=1,
        embed='timeF',
        freq='h',
        dropout=0.1,
        activation='gelu',
        diffusion_steps=20,  # 小步数加速测试
        beta_schedule='cosine',
        cond_dim=32,
        unet_channels=[16, 32],  # 小通道数
        n_samples=5,  # 少量采样
        parameterization='v',
        # 训练配置
        training_mode='end_to_end',
        train_epochs=5,
        warmup_epochs=2,
        learning_rate=1e-3,
        batch_size=4,
        # 采样配置
        use_ddim=True,
        ddim_steps=5,
        chunk_size=2,
        # AMP
        use_amp=False,
        # 时序损失
        use_ts_loss=False,
        # 其他
        use_gpu=False,
        use_multi_gpu=False,
        devices='0',
        model='iTransformerDiffusionDirect',
        data='ETTh1',
        root_path='./dataset/ETT-small/',
        data_path='ETTh1.csv',
        features='M',
        target='OT',
        scale=True,
        inverse=False,
        seasonal_patterns=None,
        augmentation_ratio=0,
        checkpoints='./test_checkpoints/',
        patience=3,
        des='test',
        itr=1,
        num_workers=0,
        model_id='test',
        lradj='type1',
        pct_start=0.3,
    )


class TestFullTrainingStep:
    """测试 1: 完整训练步骤"""

    def test_full_training_step(self):
        from exp.exp_diffusion_forecast import Exp_Diffusion_Forecast
        from data_provider.data_factory import data_provider

        print("\n  [1/5] 初始化模型和数据...")
        args = get_integration_test_config()
        exp = Exp_Diffusion_Forecast(args)

        # 获取数据加载器
        train_data, train_loader = data_provider(args, 'train')

        # 只取一个batch测试
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))

        # 确保数据类型为 float32
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        print(f"  数据形状: x={batch_x.shape}, y={batch_y.shape}")

        # 获取模型
        model = exp.model
        model.train()

        # 获取优化器
        optimizer = exp._select_optimizer_end_to_end()

        print("\n  [2/5] 测试 warmup 阶段训练...")
        # Warmup 阶段 (epoch 0)
        epoch = 0
        alpha, beta = exp._get_loss_weights(epoch)
        assert alpha >= 0.95, f"warmup 阶段 alpha 应接近 1.0，实际为 {alpha}"

        # 前向传播
        outputs = batch_y[:, -args.pred_len:, :]
        loss, loss_dict = model.forward_loss(
            batch_x, batch_x_mark, outputs, stage='warmup'
        )

        assert torch.isfinite(loss), "warmup 阶段损失包含 NaN/Inf"
        assert 'loss_mse' in loss_dict, "warmup 阶段应返回 loss_mse"

        print(f"  Warmup 损失: {loss.item():.6f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print("\n  [3/5] 测试 joint 阶段训练...")
        # Joint 阶段 (epoch > warmup)
        epoch = args.warmup_epochs + 1
        alpha, beta = exp._get_loss_weights(epoch)
        assert alpha == 0.3, f"joint 阶段 alpha 应为 0.3，实际为 {alpha}"

        # 前向传播
        model.train()
        loss, loss_dict = model.forward_loss(
            batch_x, batch_x_mark, outputs, stage='joint'
        )

        assert torch.isfinite(loss), "joint 阶段损失包含 NaN/Inf"
        assert 'loss_total' in loss_dict, "joint 阶段应返回 loss_total"
        assert 'loss_mse' in loss_dict, "joint 阶段应返回 loss_mse"
        assert 'loss_diff' in loss_dict, "joint 阶段应返回 loss_diff"

        print(f"  Joint 损失: {loss.item():.6f}")
        print(f"    - MSE: {loss_dict['loss_mse']:.6f}")
        print(f"    - Diff: {loss_dict['loss_diff']:.6f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print("\n  [4/5] 验证梯度流...")
        # 检查梯度
        has_gradient = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                assert torch.isfinite(param.grad).all(), f"{name} 的梯度包含 NaN/Inf"

        assert has_gradient, "没有参数有梯度"
        print("  梯度检查通过")

        print("\n  [5/5] 完整训练步骤验证完成")
        print("✓ test_full_training_step 通过")


class TestProbabilisticPrediction:
    """测试 2: 概率预测"""

    def test_probabilistic_prediction(self):
        from exp.exp_diffusion_forecast import Exp_Diffusion_Forecast
        from data_provider.data_factory import data_provider

        print("\n  [1/4] 初始化模型和数据...")
        args = get_integration_test_config()
        exp = Exp_Diffusion_Forecast(args)

        # 获取数据
        test_data, test_loader = data_provider(args, 'test')
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))

        # 确保数据类型为 float32
        batch_x = batch_x.float()
        batch_x_mark = batch_x_mark.float()

        print(f"  数据形状: x={batch_x.shape}")

        # 获取模型并设为评估模式
        model = exp.model
        model.eval()

        print("\n  [2/4] 执行概率预测...")
        with torch.no_grad():
            mean_pred, std_pred, samples = model.predict(
                batch_x,
                batch_x_mark,
                n_samples=args.n_samples,
                use_ddim=args.use_ddim,
                ddim_steps=args.ddim_steps,
            )

        print("\n  [3/4] 验证输出形状...")
        B = batch_x.shape[0]
        expected_mean_shape = (B, args.pred_len, args.c_out)
        expected_samples_shape = (args.n_samples, B, args.pred_len, args.c_out)

        assert mean_pred.shape == expected_mean_shape, \
            f"mean_pred 形状错误: {mean_pred.shape} vs {expected_mean_shape}"
        assert std_pred.shape == expected_mean_shape, \
            f"std_pred 形状错误: {std_pred.shape} vs {expected_mean_shape}"
        assert samples.shape == expected_samples_shape, \
            f"samples 形状错误: {samples.shape} vs {expected_samples_shape}"

        print(f"  - mean_pred: {mean_pred.shape}")
        print(f"  - std_pred: {std_pred.shape}")
        print(f"  - samples: {samples.shape}")

        print("\n  [4/4] 验证数值有效性...")
        # 验证所有输出都是有限值
        assert torch.isfinite(mean_pred).all(), "mean_pred 包含 NaN/Inf"
        assert torch.isfinite(std_pred).all(), "std_pred 包含 NaN/Inf"
        assert torch.isfinite(samples).all(), "samples 包含 NaN/Inf"

        # 验证标准差为正
        assert (std_pred >= 0).all(), "std_pred 应该全为非负值"
        assert (std_pred > 0).any(), "std_pred 应该有正值（表示有不确定性）"

        print(f"  - mean_pred 范围: [{mean_pred.min().item():.3f}, {mean_pred.max().item():.3f}]")
        print(f"  - std_pred 范围: [{std_pred.min().item():.3f}, {std_pred.max().item():.3f}]")
        print(f"  - samples 范围: [{samples.min().item():.3f}, {samples.max().item():.3f}]")

        print("\n✓ test_probabilistic_prediction 通过")


if __name__ == '__main__':
    print("=" * 60)
    print("阶段 5：集成测试")
    print("=" * 60)

    # 运行所有测试
    test_classes = [
        TestFullTrainingStep(),
        TestProbabilisticPrediction(),
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
                    print(f"\n✗ {method_name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1

    print("=" * 60)
    print(f"阶段 5 测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        exit(1)
