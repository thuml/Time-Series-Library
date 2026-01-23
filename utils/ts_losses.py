"""
时序感知复合损失函数模块

用于 iTransformerDiffusion 等时序预测模型，提供：
- 点级损失 (MSE/MAE)
- 趋势损失 (一阶差分)
- 频域损失 (FFT 幅度谱)
- 相关性损失 (变量间相关矩阵)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesAwareLoss(nn.Module):
    """
    时序感知复合损失函数

    结合多种损失项以捕捉时序数据的不同特性：
    - point: 点级误差 (MSE)
    - trend: 趋势/变化率 (一阶差分 MSE)
    - freq: 周期性模式 (频域幅度谱 MSE)
    - corr: 变量间相关性 (相关矩阵 MSE)

    Args:
        lambda_point: 点级损失权重 (默认 1.0)
        lambda_trend: 趋势损失权重 (默认 0.1)
        lambda_freq: 频域损失权重 (默认 0.1)
        lambda_corr: 相关性损失权重 (默认 0.05)
    """

    def __init__(self, lambda_point=1.0, lambda_trend=0.1,
                 lambda_freq=0.1, lambda_corr=0.05):
        super().__init__()
        self.lambdas = {
            'point': lambda_point,
            'trend': lambda_trend,
            'freq': lambda_freq,
            'corr': lambda_corr
        }

    def point_loss(self, pred, target):
        """
        点级 MSE 损失

        Args:
            pred: [B, T, N] 预测值
            target: [B, T, N] 真实值
        Returns:
            scalar loss
        """
        return F.mse_loss(pred, target)

    def trend_loss(self, pred, target):
        """
        趋势损失：一阶差分的 MSE

        捕捉时序的局部变化趋势，确保预测值的变化率与真实值匹配

        Args:
            pred: [B, T, N] 预测值
            target: [B, T, N] 真实值
        Returns:
            scalar loss
        """
        # 计算一阶差分 (时间维度)
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        target_diff = target[:, 1:, :] - target[:, :-1, :]
        return F.mse_loss(pred_diff, target_diff)

    def frequency_loss(self, pred, target):
        """
        频域损失：FFT 幅度谱的 MSE

        捕捉周期性模式，确保预测的频率特性与真实值匹配

        Args:
            pred: [B, T, N] 预测值
            target: [B, T, N] 真实值
        Returns:
            scalar loss
        """
        # 在时间维度做 FFT
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)

        # 取幅度谱 (忽略相位)
        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()

        return F.mse_loss(pred_mag, target_mag)

    def correlation_loss(self, pred, target):
        """
        相关性损失：变量间相关矩阵的 MSE

        保持多变量间的相关结构，避免破坏变量间的协变关系

        Args:
            pred: [B, T, N] 预测值
            target: [B, T, N] 真实值
        Returns:
            scalar loss
        """
        def compute_corr_matrix(x):
            """计算批次内的相关矩阵"""
            # x: [B, T, N]
            B, T, N = x.shape

            # 中心化
            x_centered = x - x.mean(dim=1, keepdim=True)

            # 标准化
            x_std = x.std(dim=1, keepdim=True) + 1e-5
            x_norm = x_centered / x_std

            # 计算相关矩阵: [B, N, N]
            # corr[i,j] = (1/T) * sum_t(x_norm[t,i] * x_norm[t,j])
            corr = torch.bmm(x_norm.transpose(1, 2), x_norm) / T

            return corr

        pred_corr = compute_corr_matrix(pred)
        target_corr = compute_corr_matrix(target)

        return F.mse_loss(pred_corr, target_corr)

    def forward(self, pred, target):
        """
        计算总损失

        Args:
            pred: [B, T, N] 预测值
            target: [B, T, N] 真实值
        Returns:
            total_loss: 加权总损失
            loss_dict: 各项损失的字典（用于日志）
        """
        losses = {
            'point': self.point_loss(pred, target),
            'trend': self.trend_loss(pred, target),
            'freq': self.frequency_loss(pred, target),
            'corr': self.correlation_loss(pred, target)
        }

        # 加权求和
        total = sum(self.lambdas[k] * v for k, v in losses.items())

        # 返回损失字典（用于日志记录）
        loss_dict = {f'loss_{k}': v.item() for k, v in losses.items()}
        loss_dict['loss_total'] = total.item()

        return total, loss_dict


class CRPSLoss(nn.Module):
    """
    CRPS 损失的可微近似

    Continuous Ranked Probability Score 用于概率预测评估

    CRPS = E[|Y - X|] - 0.5 * E[|X - X'|]
    其中 Y 是真实值，X 和 X' 是独立采样

    Args:
        n_samples: 用于估计期望的采样数
    """

    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, samples, target):
        """
        计算 CRPS 损失

        Args:
            samples: [n_samples, B, T, N] 采样预测
            target: [B, T, N] 真实值
        Returns:
            crps: scalar loss
        """
        n_samples = samples.shape[0]

        # 第一项: E[|Y - X|]
        # 使用均值预测近似
        mean_pred = samples.mean(dim=0)
        term1 = torch.abs(target - mean_pred).mean()

        # 第二项: E[|X - X'|] / 2
        # 随机选取两组样本计算
        n_pairs = min(n_samples // 2, 50)
        idx1 = torch.randperm(n_samples)[:n_pairs]
        idx2 = torch.randperm(n_samples)[:n_pairs]
        term2 = torch.abs(samples[idx1] - samples[idx2]).mean() / 2

        return term1 - term2


class CalibrationLoss(nn.Module):
    """
    校准损失

    确保预测分位数包含正确比例的真实值

    Args:
        quantiles: 要校准的分位数列表
    """

    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, samples, target):
        """
        计算校准损失

        Args:
            samples: [n_samples, B, T, N] 采样预测
            target: [B, T, N] 真实值
        Returns:
            loss: 校准误差
        """
        loss = 0.0

        for q in self.quantiles:
            # 计算预测的 q 分位数
            q_pred = torch.quantile(samples, q, dim=0)

            # 真实值应有 q 比例小于 q_pred
            actual_below = (target < q_pred).float().mean()

            # 平方误差
            loss += (actual_below - q) ** 2

        return loss / len(self.quantiles)


class DiffusionLoss(nn.Module):
    """
    扩散模型专用复合损失

    结合确定性损失和概率损失

    Args:
        ts_loss_weights: TimeSeriesAwareLoss 的权重字典
        use_crps: 是否使用 CRPS 损失（推理时计算，训练时可选）
    """

    def __init__(self, ts_loss_weights=None, use_crps=False):
        super().__init__()

        if ts_loss_weights is None:
            ts_loss_weights = {
                'point': 1.0,
                'trend': 0.1,
                'freq': 0.1,
                'corr': 0.05
            }

        self.ts_loss = TimeSeriesAwareLoss(**{f'lambda_{k}': v for k, v in ts_loss_weights.items()})
        self.use_crps = use_crps

        if use_crps:
            self.crps_loss = CRPSLoss()

    def forward(self, pred, target, samples=None):
        """
        计算复合损失

        Args:
            pred: [B, T, N] 均值预测
            target: [B, T, N] 真实值
            samples: [n_samples, B, T, N] 采样预测（可选）
        Returns:
            total_loss, loss_dict
        """
        # 时序感知损失
        ts_loss, loss_dict = self.ts_loss(pred, target)

        total = ts_loss

        # 可选: CRPS 损失
        if self.use_crps and samples is not None:
            crps = self.crps_loss(samples, target)
            loss_dict['loss_crps'] = crps.item()
            total = total + 0.1 * crps

        loss_dict['loss_total'] = total.item()

        return total, loss_dict
