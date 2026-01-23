"""
分析训练时Test Loss与测试时MSE的差异

训练时Test Loss: 0.3895 (backbone确定性预测)
测试时MSE: 0.6452 (概率预测均值)

目标：找出为什么概率预测的MSE比确定性预测高65%
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from exp.exp_diffusion_forecast import Exp_Diffusion_Forecast
from utils.tools import dotdict
import argparse

# 创建配置
args = argparse.Namespace()
args.task_name = 'diffusion_forecast'
args.is_training = 0  # 测试模式
args.model_id = 'ETTh1_96'
args.model = 'iTransformerDiffusionDirect'
args.data = 'ETTh1'
args.root_path = './dataset/ETT-small/'
args.data_path = 'ETTh1.csv'
args.features = 'M'
args.target = 'OT'
args.freq = 'h'
args.checkpoints = './checkpoints/'
args.seq_len = 96
args.label_len = 48
args.pred_len = 96
args.enc_in = 7
args.dec_in = 7
args.c_out = 7
args.d_model = 128
args.d_ff = 128
args.n_heads = 8
args.e_layers = 2
args.d_layers = 1
args.factor = 3
args.dropout = 0.1
args.embed = 'timeF'
args.activation = 'gelu'
args.num_workers = 10
args.itr = 1
args.train_epochs = 30
args.batch_size = 32
args.patience = 5
args.learning_rate = 0.0001
args.des = 'Fixed_MSE'
args.loss = 'MSE'
args.lradj = 'type1'
args.use_amp = True
args.use_gpu = True
args.gpu = 0
args.use_multi_gpu = False
args.devices = '0,1,2,3'
args.p_hidden_dims = [128, 128]
args.p_hidden_layers = 2

# 扩散参数
args.diffusion_steps = 1000
args.beta_schedule = 'cosine'
args.cond_dim = 256
args.n_samples = 100
args.use_ddim = True
args.ddim_steps = 50
args.chunk_size = 10
args.parameterization = 'v'
args.warmup_epochs = 10
args.training_mode = 'end_to_end'

# 设置setting
setting = 'diffusion_forecast_ETTh1_96_iTransformerDiffusionDirect_ETTh1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Fixed_MSE_0'

print("=" * 80)
print("分析训练时Test Loss与测试时MSE的差异")
print("=" * 80)

# 创建实验
exp = Exp_Diffusion_Forecast(args)

# 加载最佳模型
checkpoint_path = f'./checkpoints/{setting}/checkpoint.pth'
exp.model.load_state_dict(torch.load(checkpoint_path))
exp.model.eval()

# 获取测试数据
test_data, test_loader = exp._get_data(flag='test')

print(f"\n加载模型: {checkpoint_path}")
print(f"测试集大小: {len(test_data)}")

# 分析第一个batch
batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
batch_x = batch_x.float().to(exp.device)
batch_y = batch_y.float().to(exp.device)
batch_x_mark = batch_x_mark.float().to(exp.device)

f_dim = -1 if args.features == 'MS' else 0
y_true = batch_y[:, -args.pred_len:, f_dim:]

print(f"\nBatch shape: {batch_x.shape}")
print(f"y_true shape: {y_true.shape}")

with torch.no_grad():
    # 1. Backbone确定性预测
    y_det, z, means, stdev = exp.model.backbone_forward(batch_x, batch_x_mark)
    mse_det = torch.nn.functional.mse_loss(y_det, y_true).item()

    print("\n" + "=" * 80)
    print("1. Backbone确定性预测")
    print("=" * 80)
    print(f"MSE (确定性): {mse_det:.6f}")

    # 2. 概率预测（简单均值）
    print("\n" + "=" * 80)
    print("2. 概率预测 - 简单均值")
    print("=" * 80)

    mean_pred_simple, std_pred, samples = exp.model.predict(
        batch_x, batch_x_mark,
        n_samples=100,
        use_ddim=True,
        ddim_steps=50,
        use_batch_sampling=True,
        chunk_size=10,
        use_mom=False  # 禁用MoM，使用简单均值
    )

    mse_prob_simple = torch.nn.functional.mse_loss(mean_pred_simple, y_true).item()
    print(f"MSE (简单均值): {mse_prob_simple:.6f}")
    print(f"平均标准差: {std_pred.mean().item():.6f}")
    print(f"vs 确定性: {(mse_prob_simple/mse_det - 1)*100:.1f}% 差异")

    # 3. 概率预测（MoM）
    print("\n" + "=" * 80)
    print("3. 概率预测 - Median-of-Means")
    print("=" * 80)

    mean_pred_mom, std_pred_mom, samples_mom = exp.model.predict(
        batch_x, batch_x_mark,
        n_samples=100,
        use_ddim=True,
        ddim_steps=50,
        use_batch_sampling=True,
        chunk_size=10,
        use_mom=True,  # 启用MoM
        mom_k=10
    )

    mse_prob_mom = torch.nn.functional.mse_loss(mean_pred_mom, y_true).item()
    print(f"MSE (MoM): {mse_prob_mom:.6f}")
    print(f"vs 简单均值: {(mse_prob_mom/mse_prob_simple - 1)*100:.1f}% 差异")
    print(f"vs 确定性: {(mse_prob_mom/mse_det - 1)*100:.1f}% 差异")

    # 4. 分析采样分布
    print("\n" + "=" * 80)
    print("4. 采样分布分析")
    print("=" * 80)

    # samples: [n_samples, B, pred_len, N]
    samples_np = samples.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    # 每个样本与真值的MSE
    sample_mses = []
    for i in range(samples_np.shape[0]):
        sample_mse = np.mean((samples_np[i] - y_true_np) ** 2)
        sample_mses.append(sample_mse)

    sample_mses = np.array(sample_mses)

    print(f"样本MSE统计:")
    print(f"  均值: {sample_mses.mean():.6f}")
    print(f"  中位数: {np.median(sample_mses):.6f}")
    print(f"  标准差: {sample_mses.std():.6f}")
    print(f"  最小: {sample_mses.min():.6f}")
    print(f"  最大: {sample_mses.max():.6f}")

    # 检查是否有异常值
    q1, q3 = np.percentile(sample_mses, [25, 75])
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    n_outliers = np.sum(sample_mses > outlier_threshold)

    print(f"\n异常值检测:")
    print(f"  Q1: {q1:.6f}, Q3: {q3:.6f}, IQR: {iqr:.6f}")
    print(f"  阈值: {outlier_threshold:.6f}")
    print(f"  异常样本数: {n_outliers}/{len(sample_mses)}")

    # 5. 对比不同采样数量
    print("\n" + "=" * 80)
    print("5. 不同采样数量的影响")
    print("=" * 80)

    for n in [10, 50, 100, 200]:
        indices = np.random.choice(samples_np.shape[0], min(n, samples_np.shape[0]), replace=False)
        subset_samples = samples_np[indices]
        subset_mean = subset_samples.mean(axis=0)
        subset_mse = np.mean((subset_mean - y_true_np) ** 2)
        print(f"n_samples={n:3d}: MSE = {subset_mse:.6f}")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"1. Backbone确定性预测是最准确的 (MSE={mse_det:.4f})")
print(f"2. 概率预测MSE更高，说明采样方差大")
print(f"3. MoM对MSE的改善: {((mse_prob_simple - mse_prob_mom)/mse_prob_simple * 100):.1f}%")
print(f"4. 概率预测vs确定性的gap: {((mse_prob_mom - mse_det)/mse_det * 100):.1f}%")
print("\n建议:")
if mse_prob_mom / mse_det > 1.5:
    print("- 概率预测MSE显著高于确定性预测")
    print("- 考虑：增加训练epochs、调整diffusion权重、或直接用确定性预测")
else:
    print("- 概率预测性能接近确定性预测")
    print("- MoM方法有效降低MSE")
