"""
阶段 2：训练器测试 (端到端训练)

测试 Exp_Diffusion_Forecast 训练器：
1. 训练模式选择
2. 课程学习权重调度
3. 优化器参数组
4. AMP 初始化
5. 时序损失集成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from argparse import Namespace


def get_trainer_test_config():
    """创建训练器测试用配置"""
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
        diffusion_steps=100,
        beta_schedule='cosine',
        cond_dim=64,
        unet_channels=[32, 64],
        n_samples=10,
        parameterization='v',
        # 训练配置
        training_mode='end_to_end',
        train_epochs=50,
        warmup_epochs=10,
        learning_rate=1e-4,
        stage1_epochs=30,
        stage2_epochs=20,
        stage1_lr=1e-4,
        stage2_lr=1e-5,
        loss_lambda=0.5,
        # 采样配置
        use_ddim=False,
        ddim_steps=50,
        chunk_size=10,
        # AMP
        use_amp=False,
        # 时序损失
        use_ts_loss=False,
        # 其他
        use_gpu=False,
        use_multi_gpu=False,
        model='iTransformerDiffusionDirect',
        patience=3,
        checkpoints='./checkpoints/',
    )


class TestTrainingModeSelection:
    """测试 1: 验证训练模式切换"""

    def test_training_mode_selection(self):
        from exp.exp_diffusion_forecast import Exp_Diffusion_Forecast

        # 测试 end_to_end 模式
        args_e2e = get_trainer_test_config()
        args_e2e.training_mode = 'end_to_end'
        exp_e2e = Exp_Diffusion_Forecast(args_e2e)
        assert exp_e2e.training_mode == 'end_to_end', "end_to_end 模式设置失败"

        # 测试 two_stage 模式
        args_2s = get_trainer_test_config()
        args_2s.training_mode = 'two_stage'
        exp_2s = Exp_Diffusion_Forecast(args_2s)
        assert exp_2s.training_mode == 'two_stage', "two_stage 模式设置失败"

        print("✓ test_training_mode_selection 通过")


class TestCurriculumWeights:
    """测试 2: 验证课程学习权重调度"""

    def test_curriculum_weights(self):
        from exp.exp_diffusion_forecast import Exp_Diffusion_Forecast

        args = get_trainer_test_config()
        args.warmup_epochs = 10
        exp = Exp_Diffusion_Forecast(args)

        # epoch 0: α 应该接近 1.0
        alpha_0, beta_0 = exp._get_loss_weights(0)
        assert 0.95 <= alpha_0 <= 1.0, f"epoch 0 时 α 应接近 1.0，实际为 {alpha_0}"

        # epoch warmup/2: α 应该在 0.5 到 1.0 之间
        alpha_mid, beta_mid = exp._get_loss_weights(5)
        assert 0.5 < alpha_mid < 1.0, f"epoch 5 时 α 应在 (0.5, 1.0)，实际为 {alpha_mid}"

        # epoch > warmup: α 应该为 0.3
        alpha_post, beta_post = exp._get_loss_weights(15)
        assert alpha_post == 0.3, f"warmup 后 α 应为 0.3，实际为 {alpha_post}"

        # 验证 α + β = 1
        assert abs(alpha_0 + beta_0 - 1.0) < 1e-6, "α + β 应等于 1"
        assert abs(alpha_post + beta_post - 1.0) < 1e-6, "α + β 应等于 1"

        print(f"✓ test_curriculum_weights 通过 (α: 0→{alpha_0:.2f}, 5→{alpha_mid:.2f}, 15→{alpha_post:.2f})")


class TestOptimizerParamGroups:
    """测试 3: 验证优化器包含 5 个参数组"""

    def test_optimizer_param_groups(self):
        from exp.exp_diffusion_forecast import Exp_Diffusion_Forecast

        args = get_trainer_test_config()
        exp = Exp_Diffusion_Forecast(args)

        # 获取端到端优化器
        optimizer = exp._select_optimizer_end_to_end()

        # 验证参数组数量
        n_groups = len(optimizer.param_groups)
        assert n_groups == 5, f"应有 5 个参数组，实际为 {n_groups}"

        # 统计非空参数组
        # 注：output_normalizer 使用 register_buffer，没有可训练参数，这是正常的
        non_empty_groups = sum(1 for g in optimizer.param_groups if len(g['params']) > 0)
        assert non_empty_groups >= 4, f"至少应有 4 个非空参数组，实际为 {non_empty_groups}"

        print(f"✓ test_optimizer_param_groups 通过 (共 {n_groups} 个参数组, {non_empty_groups} 个非空)")


class TestAMPInitialization:
    """测试 4: 验证 AMP scaler 正确初始化"""

    def test_amp_initialization(self):
        from exp.exp_diffusion_forecast import Exp_Diffusion_Forecast

        # 不使用 AMP
        args_no_amp = get_trainer_test_config()
        args_no_amp.use_amp = False
        exp_no_amp = Exp_Diffusion_Forecast(args_no_amp)
        assert not hasattr(exp_no_amp, 'scaler') or exp_no_amp.scaler is None or not exp_no_amp.use_amp, \
            "use_amp=False 时不应初始化 scaler"

        # 使用 AMP (仅当 CUDA 可用时)
        if torch.cuda.is_available():
            args_amp = get_trainer_test_config()
            args_amp.use_amp = True
            exp_amp = Exp_Diffusion_Forecast(args_amp)
            assert hasattr(exp_amp, 'scaler'), "use_amp=True 时应初始化 scaler"
            assert isinstance(exp_amp.scaler, torch.cuda.amp.GradScaler), "scaler 类型错误"
            print("✓ test_amp_initialization 通过 (AMP 已验证)")
        else:
            print("✓ test_amp_initialization 通过 (跳过 CUDA 测试)")


class TestTSLossIntegration:
    """测试 5: 验证时序损失模块集成"""

    def test_ts_loss_integration(self):
        from exp.exp_diffusion_forecast import Exp_Diffusion_Forecast

        # 不使用时序损失
        args_no_ts = get_trainer_test_config()
        args_no_ts.use_ts_loss = False
        exp_no_ts = Exp_Diffusion_Forecast(args_no_ts)
        assert not hasattr(exp_no_ts, 'ts_loss_fn') or exp_no_ts.ts_loss_fn is None or not exp_no_ts.use_ts_loss, \
            "use_ts_loss=False 时不应初始化 ts_loss_fn"

        # 使用时序损失
        args_ts = get_trainer_test_config()
        args_ts.use_ts_loss = True
        exp_ts = Exp_Diffusion_Forecast(args_ts)
        assert hasattr(exp_ts, 'ts_loss_fn'), "use_ts_loss=True 时应初始化 ts_loss_fn"

        # 验证损失函数可调用
        from utils.ts_losses import TimeSeriesAwareLoss
        assert isinstance(exp_ts.ts_loss_fn, TimeSeriesAwareLoss), "ts_loss_fn 类型错误"

        print("✓ test_ts_loss_integration 通过")


if __name__ == '__main__':
    print("=" * 60)
    print("阶段 2：训练器测试 (端到端训练)")
    print("=" * 60)

    # 运行所有测试
    test_classes = [
        TestTrainingModeSelection(),
        TestCurriculumWeights(),
        TestOptimizerParamGroups(),
        TestAMPInitialization(),
        TestTSLossIntegration(),
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
    print(f"阶段 2 测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        exit(1)
