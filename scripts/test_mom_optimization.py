"""
测试 Median-of-Means (MoM) 优化效果

对比不同的 MoM 配置对 MSE 和 CRPS 的影响。
"""
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_provider
from models.iTransformerDiffusionDirect import Model
import argparse


# 从 exp_diffusion_forecast.py 复制的评估函数
def crps_score(samples, y_true):
    """
    计算 Continuous Ranked Probability Score (CRPS)

    Args:
        samples: [n_samples, B, T, N] 预测样本
        y_true: [B, T, N] 真实值
    Returns:
        float: CRPS分数（越小越好）
    """
    # samples: [n_samples, B, T, N]
    # y_true: [B, T, N]
    n_samples = samples.shape[0]

    # 计算 |sample - y_true|
    # [n_samples, B, T, N]
    abs_diff = torch.abs(samples - y_true.unsqueeze(0))
    term1 = abs_diff.mean(dim=0)  # [B, T, N]

    # 计算 |sample_i - sample_j| for all pairs
    # 扩展维度进行成对差分
    samples_i = samples.unsqueeze(1)  # [n_samples, 1, B, T, N]
    samples_j = samples.unsqueeze(0)  # [1, n_samples, B, T, N]
    pairwise_diff = torch.abs(samples_i - samples_j)  # [n_samples, n_samples, B, T, N]
    term2 = pairwise_diff.mean(dim=(0, 1)) / 2.0  # [B, T, N]

    # CRPS = E[|X - Y|] - 0.5*E[|X - X'|]
    crps = term1 - term2

    return crps.mean().item()


def calibration_score(samples, y_true, coverage_levels=[0.5, 0.9]):
    """
    计算预测区间的校准度

    Args:
        samples: [n_samples, B, T, N] 预测样本
        y_true: [B, T, N] 真实值
        coverage_levels: list of float, 名义覆盖率
    Returns:
        list of float: 实际覆盖率
    """
    # samples: [n_samples, B, T, N]
    # y_true: [B, T, N]

    results = []
    for level in coverage_levels:
        alpha = (1 - level) / 2
        lower_quantile = alpha
        upper_quantile = 1 - alpha

        # 计算分位数
        lower = torch.quantile(samples, lower_quantile, dim=0)  # [B, T, N]
        upper = torch.quantile(samples, upper_quantile, dim=0)  # [B, T, N]

        # 检查真值是否在区间内
        in_interval = (y_true >= lower) & (y_true <= upper)
        coverage = in_interval.float().mean().item()

        results.append(coverage)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='测试 MoM 优化')

    # 基础参数
    parser.add_argument('--data', type=str, default='ETTh1', help='数据集')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='数据路径')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--n_samples', type=int, default=100, help='采样数量')
    parser.add_argument('--use_ddim', action='store_true', help='使用DDIM采样')
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM步数')
    parser.add_argument('--chunk_size', type=int, default=10, help='分块采样大小')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度')

    # 模型参数（需要与训练时一致）
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)

    # 任务相关参数
    parser.add_argument('--task_name', type=str, default='diffusion_forecast')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)
    parser.add_argument('--target', type=str, default='OT')

    # 扩散模型参数
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine')
    parser.add_argument('--parameterization', type=str, default='x0')

    args = parser.parse_args()
    return args


def test_mom_config(model, test_loader, use_mom, mom_k, args, device):
    """
    测试单个 MoM 配置

    Returns:
        dict: 包含 MSE, MAE, RMSE, CRPS, 校准度等指标
    """
    model.eval()

    preds = []
    trues = []
    crps_scores = []
    calibration_results = []

    autocast_context = torch.cuda.amp.autocast() if args.use_amp else torch.cuda.amp.autocast(enabled=False)

    with torch.no_grad():
        with autocast_context:
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)

                # 概率预测（使用指定的 MoM 配置）
                mean_pred, std_pred, samples = model.predict(
                    batch_x, batch_x_mark,
                    n_samples=args.n_samples,
                    use_ddim=args.use_ddim,
                    ddim_steps=args.ddim_steps,
                    use_batch_sampling=True,
                    chunk_size=args.chunk_size,
                    use_mom=use_mom,
                    mom_k=mom_k
                )

                f_dim = -1 if args.features == 'MS' else 0
                y_true = batch_y[:, -args.pred_len:, f_dim:]
                mean_pred = mean_pred[:, :, f_dim:]
                std_pred = std_pred[:, :, f_dim:]
                samples = samples[:, :, :, f_dim:]

                # 计算 CRPS
                batch_crps = crps_score(samples, y_true)
                crps_scores.append(batch_crps)

                # 计算校准度
                batch_calib = calibration_score(samples, y_true)
                calibration_results.append(batch_calib)

                # 存储预测和真值
                preds.append(mean_pred.cpu().numpy())
                trues.append(y_true.cpu().numpy())

                if i % 20 == 0:
                    print(f"  Testing batch {i}/{len(test_loader)}")

    # 聚合结果
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # 计算点预测指标
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(mse)

    # 计算概率预测指标
    crps = np.mean(crps_scores)

    # 计算平均校准度
    calib_50 = np.mean([c[0] for c in calibration_results])
    calib_90 = np.mean([c[1] for c in calibration_results])

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'crps': crps,
        'calib_50': calib_50,
        'calib_90': calib_90
    }


def main():
    args = parse_args()

    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("\n加载测试数据...")
    test_data, test_loader = data_provider(args, flag='test')

    # 创建模型
    print("\n创建模型...")
    model = Model(args).to(device)

    # 加载 checkpoint
    print(f"\n加载模型权重: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 测试配置列表
    test_configs = [
        {'use_mom': False, 'mom_k': None, 'name': '简单均值 (Baseline)'},
        {'use_mom': True, 'mom_k': 5, 'name': 'MoM k=5'},
        {'use_mom': True, 'mom_k': 10, 'name': 'MoM k=10 (当前)'},
        {'use_mom': True, 'mom_k': 20, 'name': 'MoM k=20'},
    ]

    print("\n" + "="*80)
    print("开始 MoM 优化测试")
    print("="*80)

    results = []

    for config in test_configs:
        print(f"\n测试配置: {config['name']}")
        print("-" * 80)

        metrics = test_mom_config(
            model, test_loader,
            use_mom=config['use_mom'],
            mom_k=config['mom_k'] if config['use_mom'] else 10,
            args=args,
            device=device
        )

        results.append({
            'config': config['name'],
            **metrics
        })

        print(f"  MSE:      {metrics['mse']:.6f}")
        print(f"  MAE:      {metrics['mae']:.6f}")
        print(f"  RMSE:     {metrics['rmse']:.6f}")
        print(f"  CRPS:     {metrics['crps']:.6f}")
        print(f"  Calib50:  {metrics['calib_50']:.4f}")
        print(f"  Calib90:  {metrics['calib_90']:.4f}")

    # 对比分析
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)

    baseline_mse = results[0]['mse']
    baseline_crps = results[0]['crps']

    print(f"\n{'配置':<20} {'MSE':<12} {'MSE改善':<10} {'CRPS':<12} {'CRPS改善':<10}")
    print("-" * 80)

    for r in results:
        mse_improve = (baseline_mse - r['mse']) / baseline_mse * 100
        crps_improve = (baseline_crps - r['crps']) / baseline_crps * 100

        print(f"{r['config']:<20} {r['mse']:.6f}    {mse_improve:>6.2f}%    "
              f"{r['crps']:.6f}    {crps_improve:>6.2f}%")

    # 找到最优配置
    best_mse = min(results, key=lambda x: x['mse'])
    best_crps = min(results, key=lambda x: x['crps'])

    print(f"\n最优 MSE 配置: {best_mse['config']} (MSE={best_mse['mse']:.6f})")
    print(f"最优 CRPS 配置: {best_crps['config']} (CRPS={best_crps['crps']:.6f})")

    # 保存结果
    output_file = 'mom_optimization_results.txt'
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MoM 优化测试结果\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'配置':<20} {'MSE':<12} {'MSE改善':<10} {'CRPS':<12} {'CRPS改善':<10}\n")
        f.write("-" * 80 + "\n")

        for r in results:
            mse_improve = (baseline_mse - r['mse']) / baseline_mse * 100
            crps_improve = (baseline_crps - r['crps']) / baseline_crps * 100

            f.write(f"{r['config']:<20} {r['mse']:.6f}    {mse_improve:>6.2f}%    "
                   f"{r['crps']:.6f}    {crps_improve:>6.2f}%\n")

        f.write(f"\n最优 MSE 配置: {best_mse['config']} (MSE={best_mse['mse']:.6f})\n")
        f.write(f"最优 CRPS 配置: {best_crps['config']} (CRPS={best_crps['crps']:.6f})\n")

    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
