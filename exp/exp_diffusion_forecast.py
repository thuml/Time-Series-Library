"""
Experiment class for iTransformer + CRD-Net two-stage training.

Training Strategy:
- Stage 1 (Warmup): Train iTransformer backbone with MSE loss
- Stage 2 (Joint): Train diffusion with frozen encoder, combined loss

Evaluation:
- Point prediction metrics: MSE, MAE, RMSE
- Probabilistic metrics: CRPS, Calibration
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def crps_score(samples, y_true):
    """
    Compute Continuous Ranked Probability Score (CRPS).

    Args:
        samples: [n_samples, B, pred_len, N] sampled predictions
        y_true: [B, pred_len, N] ground truth
    Returns:
        crps: scalar CRPS value
    """
    n_samples = samples.shape[0]

    # Sort samples
    samples_sorted, _ = torch.sort(samples, dim=0)

    # Compute CRPS using numerical integration
    crps = 0.0
    for i in range(n_samples):
        # Indicator function: 1 if sample <= y_true
        indicator = (samples_sorted[i] <= y_true).float()
        # Empirical CDF at this sample
        ecdf = (i + 1) / n_samples
        # Add to CRPS
        crps += (indicator - ecdf) ** 2

    crps = crps.mean() / n_samples
    return crps.item()


def calibration_score(samples, y_true, coverage_levels=[0.5, 0.9]):
    """
    Compute calibration (empirical coverage of prediction intervals).

    Args:
        samples: [n_samples, B, pred_len, N]
        y_true: [B, pred_len, N]
        coverage_levels: list of nominal coverage levels
    Returns:
        dict of coverage level -> empirical coverage
    """
    results = {}
    n_samples = samples.shape[0]

    for level in coverage_levels:
        alpha = 1 - level
        lower_idx = int(n_samples * alpha / 2)
        upper_idx = int(n_samples * (1 - alpha / 2))

        samples_sorted, _ = torch.sort(samples, dim=0)
        lower = samples_sorted[lower_idx]
        upper = samples_sorted[upper_idx]

        # Check if true values fall within interval
        within = ((y_true >= lower) & (y_true <= upper)).float().mean()
        results[f'coverage_{int(level*100)}'] = within.item()

    return results


class EarlyStoppingWithSuffix:
    """EarlyStopping with custom checkpoint suffix support."""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path, suffix=''):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, suffix)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, suffix)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, suffix=''):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path + '/checkpoint' + suffix + '.pth')
        self.val_loss_min = val_loss

    def reset(self):
        """Reset for next stage."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf


class Exp_Diffusion_Forecast(Exp_Basic):
    """
    训练器：支持两阶段训练和端到端联合训练两种模式

    训练模式:
    - 'two_stage': 原有两阶段训练（Stage 1: Backbone, Stage 2: Diffusion）
    - 'end_to_end': 端到端联合训练（推荐，梯度连通）

    支持 AMP (Automatic Mixed Precision) 以节省显存
    """

    def __init__(self, args):
        super(Exp_Diffusion_Forecast, self).__init__(args)

        # 训练模式: 'two_stage' 或 'end_to_end'
        self.training_mode = getattr(args, 'training_mode', 'end_to_end')

        # 两阶段训练配置（用于 two_stage 模式）
        self.stage1_epochs = getattr(args, 'stage1_epochs', 30)
        self.stage2_epochs = getattr(args, 'stage2_epochs', 20)
        self.stage1_lr = getattr(args, 'stage1_lr', 1e-4)
        self.stage2_lr = getattr(args, 'stage2_lr', 1e-5)
        self.loss_lambda = getattr(args, 'loss_lambda', 0.5)

        # 端到端训练配置
        self.train_epochs = getattr(args, 'train_epochs', 50)
        self.warmup_epochs = getattr(args, 'warmup_epochs', 10)
        self.learning_rate = getattr(args, 'learning_rate', 1e-4)

        # 概率预测配置
        self.n_samples = getattr(args, 'n_samples', 100)
        self.use_ddim = getattr(args, 'use_ddim', False)
        self.ddim_steps = getattr(args, 'ddim_steps', 50)
        self.chunk_size = getattr(args, 'chunk_size', 10)

        # AMP (Automatic Mixed Precision) support
        self.use_amp = getattr(args, 'use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("AMP (Automatic Mixed Precision) enabled - 显存节省 30-50%")

        # 时序感知损失
        self.use_ts_loss = getattr(args, 'use_ts_loss', False)
        if self.use_ts_loss:
            from utils.ts_losses import TimeSeriesAwareLoss
            self.ts_loss_fn = TimeSeriesAwareLoss(
                lambda_point=1.0,
                lambda_trend=0.1,
                lambda_freq=0.1,
                lambda_corr=0.05
            )
            print("时序感知损失已启用")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer_stage1(self):
        """Optimizer for stage 1: full backbone training."""
        # Only train backbone parameters
        backbone_params = list(self.model.enc_embedding.parameters()) + \
                          list(self.model.encoder.parameters()) + \
                          list(self.model.projection.parameters())
        optimizer = optim.AdamW(backbone_params, lr=self.stage1_lr, weight_decay=0.01)
        return optimizer

    def _select_optimizer_stage2(self):
        """Optimizer for stage 2: grouped learning rates."""
        # Different learning rates for different components
        param_groups = [
            {'params': self.model.projection.parameters(), 'lr': self.stage2_lr},
            {'params': self.model.denoise_net.parameters(), 'lr': self.stage2_lr * 10},
            {'params': self.model.output_normalizer.parameters(), 'lr': self.stage2_lr * 10},
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        return optimizer

    def _select_optimizer_end_to_end(self):
        """
        端到端训练优化器：所有参数联合优化

        使用分组学习率，但不冻结任何参数
        """
        param_groups = [
            {'params': self.model.enc_embedding.parameters(), 'lr': self.learning_rate},
            {'params': self.model.encoder.parameters(), 'lr': self.learning_rate},
            {'params': self.model.projection.parameters(), 'lr': self.learning_rate},
            {'params': self.model.denoise_net.parameters(), 'lr': self.learning_rate},
            {'params': self.model.output_normalizer.parameters(), 'lr': self.learning_rate},
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        return optimizer

    def _get_loss_weights(self, epoch):
        """
        课程学习权重调度（修复版）

        **问题发现**：
        - 原策略：前期α从1.0降到0.5，后期α=0.3（70% Diffusion）
        - 实验观察：Epoch 1 (α=1.0) 性能最佳，后续α降低反而MSE上升
        - 根本原因：MSE损失权重过低（0.3）导致点预测性能下降

        **修复策略**（基于SimDiff和调研发现）：
        - 固定α=0.8（80% MSE + 20% Diffusion）
        - 确保点预测性能优先，同时学习不确定性建模
        - 扩散模型的MSE问题是已知的trade-off，不应过度优化分布

        Args:
            epoch: 当前 epoch
        Returns:
            alpha: MSE 损失权重（固定0.8）
            beta: 扩散损失权重（固定0.2）
        """
        # 修复版：固定α=0.8，MSE为主
        alpha = 0.8
        beta = 0.2

        # ===== 原有策略（已废弃，保留以供参考） =====
        # if epoch < self.warmup_epochs:
        #     # Warmup 阶段: MSE 权重从 1.0 线性降到 0.5
        #     alpha = 1.0 - epoch / self.warmup_epochs * 0.5
        # else:
        #     # 联合阶段: 固定 30% MSE + 70% Diffusion
        #     alpha = 0.3
        # beta = 1.0 - alpha
        # ==========================================

        return alpha, beta

    def vali(self, vali_data, vali_loader, stage='warmup'):
        """
        验证损失计算（修复版）

        使用 backbone 的确定性预测 MSE 作为验证损失，而非 forward_loss 的混合损失。
        这确保 early stopping 基于真实的点预测性能，而非包含 diffusion 训练损失的混合指标。

        修复前问题：
        - forward_loss 返回 lambda*MSE + (1-lambda)*diffusion_loss
        - diffusion_loss 是训练用的噪声预测误差，数值量级不对（600+）
        - 导致 early stopping 在 Epoch 1 就触发

        修复后：
        - 只计算 backbone 确定性预测的 MSE
        - 数值量级合理（0.3-0.6）
        - early stopping 基于真实的点预测性能
        """
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Get ground truth for prediction horizon
                f_dim = -1 if self.args.features == 'MS' else 0
                y_true = batch_y[:, -self.args.pred_len:, f_dim:]

                # 使用 backbone 确定性预测计算 MSE（修复关键点）
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        y_det, z, means, stdev = self.model.backbone_forward(batch_x, batch_x_mark)
                        loss_mse = F.mse_loss(y_det, y_true)
                else:
                    y_det, z, means, stdev = self.model.backbone_forward(batch_x, batch_x_mark)
                    loss_mse = F.mse_loss(y_det, y_true)

                total_loss.append(loss_mse.item())

        self.model.train()
        return np.average(total_loss)

    def train_stage1(self, setting, train_loader, vali_loader, test_loader, path):
        """
        Stage 1: Warmup training for iTransformer backbone.
        Supports AMP for memory efficiency.
        """
        print("=" * 50)
        print("Stage 1: Backbone Warmup Training")
        if self.use_amp:
            print("(AMP enabled)")
        print("=" * 50)

        train_steps = len(train_loader)
        early_stopping = EarlyStoppingWithSuffix(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer_stage1()
        scheduler = CosineAnnealingLR(optimizer, T_max=self.stage1_epochs)

        for epoch in range(self.stage1_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Get ground truth
                f_dim = -1 if self.args.features == 'MS' else 0
                y_true = batch_y[:, -self.args.pred_len:, f_dim:]

                # Forward pass with AMP
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss, loss_dict = self.model.forward_loss(batch_x, batch_x_mark, y_true, stage='warmup')
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print(f"\tStage1 iters: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss, loss_dict = self.model.forward_loss(batch_x, batch_x_mark, y_true, stage='warmup')
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print(f"\tStage1 iters: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

            scheduler.step()

            print(f"Stage1 Epoch: {epoch+1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(None, vali_loader, stage='warmup')
            test_loss = self.vali(None, test_loader, stage='warmup')

            print(f"Stage1 Epoch: {epoch+1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path, suffix='_stage1')
            if early_stopping.early_stop:
                print("Stage 1 Early stopping")
                break

        # Load best stage 1 model
        best_model_path = path + '/checkpoint_stage1.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best stage 1 model from {best_model_path}")

    def train_stage2(self, setting, train_loader, vali_loader, test_loader, path):
        """
        Stage 2: Joint training with diffusion.
        Freeze encoder, train diffusion + projection.
        """
        print("=" * 50)
        print("Stage 2: Joint Training (Diffusion)")
        if self.use_amp:
            print("(AMP enabled)")
        print("=" * 50)

        # Freeze encoder
        self.model.freeze_encoder()
        print("Encoder frozen, training diffusion components...")

        train_steps = len(train_loader)
        early_stopping = EarlyStoppingWithSuffix(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer_stage2()
        scheduler = CosineAnnealingLR(optimizer, T_max=self.stage2_epochs)

        for epoch in range(self.stage2_epochs):
            iter_count = 0
            train_loss = []
            loss_mse_list = []
            loss_diff_list = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Get ground truth
                f_dim = -1 if self.args.features == 'MS' else 0
                y_true = batch_y[:, -self.args.pred_len:, f_dim:]

                # Forward pass with optional AMP
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss, loss_dict = self.model.forward_loss(batch_x, batch_x_mark, y_true, stage='joint')
                    train_loss.append(loss.item())
                    loss_mse_list.append(loss_dict['loss_mse'])
                    loss_diff_list.append(loss_dict['loss_diff'])

                    if (i + 1) % 100 == 0:
                        print(f"\tStage2 iters: {i+1}, epoch: {epoch+1} | "
                              f"loss: {loss.item():.7f} mse: {loss_dict['loss_mse']:.7f} diff: {loss_dict['loss_diff']:.7f}")

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss, loss_dict = self.model.forward_loss(batch_x, batch_x_mark, y_true, stage='joint')
                    train_loss.append(loss.item())
                    loss_mse_list.append(loss_dict['loss_mse'])
                    loss_diff_list.append(loss_dict['loss_diff'])

                    if (i + 1) % 100 == 0:
                        print(f"\tStage2 iters: {i+1}, epoch: {epoch+1} | "
                              f"loss: {loss.item():.7f} mse: {loss_dict['loss_mse']:.7f} diff: {loss_dict['loss_diff']:.7f}")

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

            scheduler.step()

            print(f"Stage2 Epoch: {epoch+1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            avg_mse = np.average(loss_mse_list)
            avg_diff = np.average(loss_diff_list)
            vali_loss = self.vali(None, vali_loader, stage='joint')
            test_loss = self.vali(None, test_loader, stage='joint')

            print(f"Stage2 Epoch: {epoch+1} | Train Loss: {train_loss:.7f} (MSE: {avg_mse:.7f}, Diff: {avg_diff:.7f}) "
                  f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path, suffix='_stage2')
            if early_stopping.early_stop:
                print("Stage 2 Early stopping")
                break

        # Load best stage 2 model
        best_model_path = path + '/checkpoint_stage2.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best stage 2 model from {best_model_path}")

    def train_end_to_end(self, setting, train_loader, vali_loader, test_loader, path):
        """
        端到端联合训练（推荐）

        与两阶段训练的区别：
        1. Backbone 和 Diffusion 同时训练，梯度连通
        2. 使用课程学习：前期 MSE 为主，后期 Diffusion 为主
        3. 不冻结任何参数
        """
        print("=" * 50)
        print("端到端联合训练 (End-to-End Training)")
        if self.use_amp:
            print("(AMP enabled)")
        if self.use_ts_loss:
            print("(时序感知损失 enabled)")
        print("=" * 50)

        train_steps = len(train_loader)
        early_stopping = EarlyStoppingWithSuffix(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer_end_to_end()
        scheduler = CosineAnnealingLR(optimizer, T_max=self.train_epochs)

        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []
            loss_mse_list = []
            loss_diff_list = []

            # 获取当前 epoch 的损失权重
            alpha, beta = self._get_loss_weights(epoch)

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Get ground truth
                f_dim = -1 if self.args.features == 'MS' else 0
                y_true = batch_y[:, -self.args.pred_len:, f_dim:]

                # Forward pass with optional AMP
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # 获取原始损失
                        loss, loss_dict = self.model.forward_loss(
                            batch_x, batch_x_mark, y_true, stage='joint'
                        )
                        loss_mse = loss_dict['loss_mse']
                        loss_diff = loss_dict['loss_diff']

                    train_loss.append(loss.item())
                    loss_mse_list.append(loss_mse)
                    loss_diff_list.append(loss_diff)

                    if (i + 1) % 100 == 0:
                        print(f"\titers: {i+1}, epoch: {epoch+1} | "
                              f"loss: {loss.item():.7f} mse: {loss_mse:.7f} diff: {loss_diff:.7f} "
                              f"(α={alpha:.2f}, β={beta:.2f})")

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # 获取原始损失
                    loss, loss_dict = self.model.forward_loss(
                        batch_x, batch_x_mark, y_true, stage='joint'
                    )
                    loss_mse = loss_dict['loss_mse']
                    loss_diff = loss_dict['loss_diff']

                    train_loss.append(loss.item())
                    loss_mse_list.append(loss_mse)
                    loss_diff_list.append(loss_diff)

                    if (i + 1) % 100 == 0:
                        print(f"\titers: {i+1}, epoch: {epoch+1} | "
                              f"loss: {loss.item():.7f} mse: {loss_mse:.7f} diff: {loss_diff:.7f} "
                              f"(α={alpha:.2f}, β={beta:.2f})")

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

            scheduler.step()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss_avg = np.average(train_loss)
            avg_mse = np.average(loss_mse_list)
            avg_diff = np.average(loss_diff_list)
            vali_loss = self.vali(None, vali_loader, stage='joint')
            test_loss = self.vali(None, test_loader, stage='joint')

            print(f"Epoch: {epoch+1} | Train: {train_loss_avg:.7f} (MSE: {avg_mse:.7f}, Diff: {avg_diff:.7f}) "
                  f"Vali: {vali_loss:.7f} Test: {test_loss:.7f} (α={alpha:.2f})")

            early_stopping(vali_loss, self.model, path, suffix='')
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load best model
        best_model_path = path + '/checkpoint.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")

    def train(self, setting):
        """
        训练入口：根据 training_mode 选择训练方式

        - 'end_to_end': 端到端联合训练（推荐）
        - 'two_stage': 两阶段训练（原有方式）
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        if self.training_mode == 'end_to_end':
            # 端到端联合训练（推荐）
            print(f"\n使用端到端训练模式 (epochs={self.train_epochs}, warmup={self.warmup_epochs})")
            self.train_end_to_end(setting, train_loader, vali_loader, test_loader, path)
        else:
            # 两阶段训练（原有方式）
            print(f"\n使用两阶段训练模式 (stage1={self.stage1_epochs}, stage2={self.stage2_epochs})")
            self.train_stage1(setting, train_loader, vali_loader, test_loader, path)
            self.train_stage2(setting, train_loader, vali_loader, test_loader, path)

            # 两阶段训练时，保存最终模型
            final_model_path = path + '/checkpoint.pth'
            torch.save(self.model.state_dict(), final_model_path)
            print(f"Final model saved to {final_model_path}")

        print(f"\nTotal training time: {time.time() - time_now:.2f}s")

        return self.model

    def test(self, setting, test=0):
        """
        Test with probabilistic predictions.
        """
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('Loading model...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        pred_stds = []
        all_samples = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        crps_scores = []
        calibration_results = []

        # Use AMP autocast for inference to save memory
        autocast_context = torch.cuda.amp.autocast() if self.use_amp else torch.cuda.amp.autocast(enabled=False)

        with torch.no_grad():
            with autocast_context:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    # Probabilistic prediction with batch sampling
                    mean_pred, std_pred, samples = self.model.predict(
                        batch_x, batch_x_mark,
                        n_samples=self.n_samples,
                        use_ddim=self.use_ddim,
                        ddim_steps=self.ddim_steps,
                        use_batch_sampling=True,
                        chunk_size=self.chunk_size
                    )

                    f_dim = -1 if self.args.features == 'MS' else 0
                    y_true = batch_y[:, -self.args.pred_len:, f_dim:]
                    mean_pred = mean_pred[:, :, f_dim:]
                    std_pred = std_pred[:, :, f_dim:]
                    samples = samples[:, :, :, f_dim:]

                    # Compute CRPS for this batch
                    batch_crps = crps_score(samples, y_true)
                    crps_scores.append(batch_crps)

                    # Compute calibration
                    batch_calib = calibration_score(samples, y_true)
                    calibration_results.append(batch_calib)

                    # Store results
                    preds.append(mean_pred.cpu().numpy())
                    trues.append(y_true.cpu().numpy())
                    pred_stds.append(std_pred.cpu().numpy())

                    if i % 20 == 0:
                        print(f"Testing batch {i}, CRPS: {batch_crps:.6f}")

                        # Visualize with uncertainty
                        if i == 0:
                            input_np = batch_x.detach().cpu().numpy()
                            pred_np = mean_pred.detach().cpu().numpy()
                            std_np = std_pred.detach().cpu().numpy()
                            true_np = y_true.detach().cpu().numpy()

                            # Save visualization data
                            np.savez(
                                os.path.join(folder_path, f'sample_{i}.npz'),
                                input=input_np[0],
                                pred=pred_np[0],
                                std=std_np[0],
                                true=true_np[0]
                            )

        # Aggregate results
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        pred_stds = np.concatenate(pred_stds, axis=0)

        print('Test shape:', preds.shape, trues.shape)

        # Point prediction metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        # Probabilistic metrics
        avg_crps = np.mean(crps_scores)
        avg_calib_50 = np.mean([c['coverage_50'] for c in calibration_results])
        avg_calib_90 = np.mean([c['coverage_90'] for c in calibration_results])
        avg_sharpness = np.mean(pred_stds)

        print("=" * 50)
        print("Test Results:")
        print("=" * 50)
        print(f"Point Metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        print(f"Prob Metrics  - CRPS: {avg_crps:.6f}")
        print(f"Calibration   - 50%: {avg_calib_50:.4f}, 90%: {avg_calib_90:.4f}")
        print(f"Sharpness     - Avg Std: {avg_sharpness:.6f}")
        print("=" * 50)

        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Write to file
        f = open("result_diffusion_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write(f'Point: mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}\n')
        f.write(f'Prob: crps:{avg_crps:.6f}, calib_50:{avg_calib_50:.4f}, calib_90:{avg_calib_90:.4f}, sharpness:{avg_sharpness:.6f}\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'prob_metrics.npy', np.array([avg_crps, avg_calib_50, avg_calib_90, avg_sharpness]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'pred_std.npy', pred_stds)
        np.save(folder_path + 'true.npy', trues)

        return mse, mae, avg_crps
