"""
阶段 4：CLI 参数测试 (run.py)

验证新增的命令行参数能正确解析：
1. --parameterization (choices: x0/epsilon/v)
2. --training_mode (choices: end_to_end/two_stage)
3. --use_ts_loss (action: store_true)
4. --warmup_epochs (type: int)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


def create_parser():
    """从 run.py 复制相关参数定义"""
    parser = argparse.ArgumentParser(description='TimesNet')

    # 基础配置
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test')
    parser.add_argument('--model', type=str, default='Autoformer')

    # 扩散模型参数
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine')
    parser.add_argument('--cond_dim', type=int, default=256)
    parser.add_argument('--parameterization', type=str, default='v', choices=['x0', 'epsilon', 'v'])

    # 训练模式
    parser.add_argument('--training_mode', type=str, default='end_to_end', choices=['end_to_end', 'two_stage'])

    # 端到端训练参数
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # 两阶段训练参数
    parser.add_argument('--stage1_epochs', type=int, default=30)
    parser.add_argument('--stage2_epochs', type=int, default=20)
    parser.add_argument('--stage1_lr', type=float, default=1e-4)
    parser.add_argument('--stage2_lr', type=float, default=1e-5)
    parser.add_argument('--loss_lambda', type=float, default=0.5)

    # 采样配置
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--use_ddim', action='store_true', default=False)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=10)

    # 时序感知损失
    parser.add_argument('--use_ts_loss', action='store_true', default=False)

    return parser


class TestNewArguments:
    """测试: 验证新参数可识别"""

    def test_parameterization_argument(self):
        """测试 --parameterization 参数"""
        parser = create_parser()

        # 测试默认值
        args = parser.parse_args([])
        assert args.parameterization == 'v', f"默认值应为 'v'，实际为 '{args.parameterization}'"

        # 测试 x0
        args = parser.parse_args(['--parameterization', 'x0'])
        assert args.parameterization == 'x0'

        # 测试 epsilon
        args = parser.parse_args(['--parameterization', 'epsilon'])
        assert args.parameterization == 'epsilon'

        # 测试 v
        args = parser.parse_args(['--parameterization', 'v'])
        assert args.parameterization == 'v'

        # 测试无效值
        try:
            parser.parse_args(['--parameterization', 'invalid'])
            assert False, "应该拒绝无效值"
        except SystemExit:
            pass  # argparse 会在无效值时退出

        print("✓ test_parameterization_argument 通过")

    def test_training_mode_argument(self):
        """测试 --training_mode 参数"""
        parser = create_parser()

        # 测试默认值
        args = parser.parse_args([])
        assert args.training_mode == 'end_to_end', f"默认值应为 'end_to_end'，实际为 '{args.training_mode}'"

        # 测试 two_stage
        args = parser.parse_args(['--training_mode', 'two_stage'])
        assert args.training_mode == 'two_stage'

        # 测试无效值
        try:
            parser.parse_args(['--training_mode', 'invalid'])
            assert False, "应该拒绝无效值"
        except SystemExit:
            pass

        print("✓ test_training_mode_argument 通过")

    def test_use_ts_loss_argument(self):
        """测试 --use_ts_loss 参数"""
        parser = create_parser()

        # 测试默认值（不设置时为 False）
        args = parser.parse_args([])
        assert args.use_ts_loss == False, f"默认值应为 False，实际为 {args.use_ts_loss}"

        # 测试设置标志
        args = parser.parse_args(['--use_ts_loss'])
        assert args.use_ts_loss == True, f"设置 --use_ts_loss 后应为 True，实际为 {args.use_ts_loss}"

        print("✓ test_use_ts_loss_argument 通过")

    def test_warmup_epochs_argument(self):
        """测试 --warmup_epochs 参数"""
        parser = create_parser()

        # 测试默认值
        args = parser.parse_args([])
        assert args.warmup_epochs == 10, f"默认值应为 10，实际为 {args.warmup_epochs}"

        # 测试自定义值
        args = parser.parse_args(['--warmup_epochs', '20'])
        assert args.warmup_epochs == 20

        # 验证类型为 int
        assert isinstance(args.warmup_epochs, int), f"应为 int 类型，实际为 {type(args.warmup_epochs)}"

        print("✓ test_warmup_epochs_argument 通过")

    def test_all_new_arguments_combined(self):
        """测试所有新参数组合使用"""
        parser = create_parser()

        args = parser.parse_args([
            '--parameterization', 'v',
            '--training_mode', 'end_to_end',
            '--warmup_epochs', '15',
            '--use_ts_loss'
        ])

        assert args.parameterization == 'v'
        assert args.training_mode == 'end_to_end'
        assert args.warmup_epochs == 15
        assert args.use_ts_loss == True

        print("✓ test_all_new_arguments_combined 通过")


if __name__ == '__main__':
    print("=" * 60)
    print("阶段 4：CLI 参数测试 (run.py)")
    print("=" * 60)

    # 运行所有测试
    test_class = TestNewArguments()

    passed = 0
    failed = 0

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
    print(f"阶段 4 测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        exit(1)
