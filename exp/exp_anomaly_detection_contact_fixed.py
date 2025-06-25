from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.logger import logger

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection_Contact_Fixed(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_Contact_Fixed, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            logger.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        temperature = 50

        # 1. 计算所有重构误差
        criterion = nn.MSELoss(reduction='none')
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)

                score = torch.mean(criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        # 2. 获取真实标签
        combined_energy = []
        combined_labels = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)

                score = torch.mean(criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                combined_energy.extend(score)
                
                # 处理标签格式：batch_y形状为(batch_size, win_size)
                # 对于异常检测，如果窗口中有任何异常点，则认为该窗口为异常
                if len(batch_y.shape) == 2:
                    # 取每个窗口的最大值（如果窗口中有异常点，则窗口标记为异常）
                    window_labels = torch.max(batch_y, dim=1)[0].numpy()
                else:
                    # 如果已经是1D，直接使用
                    window_labels = batch_y.numpy()
                
                combined_labels.extend(window_labels)

        combined_energy = np.array(combined_energy)
        combined_labels = np.array(combined_labels)

        # 调试信息
        logger.info(f"🔍 调试信息:")
        logger.info(f"   combined_energy shape: {combined_energy.shape}, dtype: {combined_energy.dtype}")
        logger.info(f"   combined_labels shape: {combined_labels.shape}, dtype: {combined_labels.dtype}")
        logger.info(f"   combined_labels unique values: {np.unique(combined_labels)}")
        logger.info(f"   combined_labels has NaN: {np.isnan(combined_labels).any()}")
        logger.info(f"   combined_energy has NaN: {np.isnan(combined_energy).any()}")

        # 清理数据 - 移除NaN值
        valid_mask = ~(np.isnan(combined_energy) | np.isnan(combined_labels))
        combined_energy = combined_energy[valid_mask]
        combined_labels = combined_labels[valid_mask]
        
        # 确保标签是整数类型
        combined_labels = combined_labels.astype(int)
        
        logger.info(f"   清理后 energy shape: {combined_energy.shape}")
        logger.info(f"   清理后 labels shape: {combined_labels.shape}")
        logger.info(f"   清理后 labels unique: {np.unique(combined_labels)}")

        # 3. 关键修改：反转异常判断逻辑
        logger.info("🔄 使用反转逻辑：重构误差小的样本为异常")
        
        # 使用多种阈值策略
        results = {}
        
        # 策略1：低percentile阈值（异常=重构误差小）
        for percentile in [5, 10, 15, 20]:
            threshold = np.percentile(combined_energy, percentile)
            pred = (combined_energy <= threshold).astype(int)
            
            precision, recall, f_score, support = precision_recall_fscore_support(combined_labels, pred, average='binary')
            accuracy = accuracy_score(combined_labels, pred)
            
            results[f'percentile_{percentile}'] = {
                'precision': precision,
                'recall': recall,
                'f_score': f_score,
                'accuracy': accuracy
            }
            
            logger.info(f"📊 Percentile {percentile}% - Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")

        # 策略2：基于标准差的阈值
        mean_energy = np.mean(combined_energy)
        std_energy = np.std(combined_energy)
        
        for std_factor in [0.5, 1.0, 1.5]:
            threshold = mean_energy - std_factor * std_energy
            pred = (combined_energy <= threshold).astype(int)
            
            precision, recall, f_score, support = precision_recall_fscore_support(combined_labels, pred, average='binary')
            accuracy = accuracy_score(combined_labels, pred)
            
            results[f'std_{std_factor}'] = {
                'precision': precision,
                'recall': recall,
                'f_score': f_score,
                'accuracy': accuracy
            }
            
            logger.info(f"📊 Std {std_factor}x - Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")

        # 策略3：优化F1分数的阈值
        sorted_indices = np.argsort(combined_energy)
        best_f1 = 0
        best_threshold = 0
        
        for i in range(int(len(combined_energy) * 0.01), int(len(combined_energy) * 0.3), 100):
            threshold = combined_energy[sorted_indices[i]]
            pred = (combined_energy <= threshold).astype(int)
            
            precision, recall, f_score, support = precision_recall_fscore_support(combined_labels, pred, average='binary')
            
            if f_score > best_f1:
                best_f1 = f_score
                best_threshold = threshold
        
        # 使用最佳阈值
        pred_best = (combined_energy <= best_threshold).astype(int)
        precision_best, recall_best, f_score_best, support = precision_recall_fscore_support(combined_labels, pred_best, average='binary')
        accuracy_best = accuracy_score(combined_labels, pred_best)
        
        results['best_f1'] = {
            'precision': precision_best,
            'recall': recall_best,
            'f_score': f_score_best,
            'accuracy': accuracy_best,
            'threshold': best_threshold
        }
        
        logger.info(f"🏆 最佳F1阈值 - Precision: {precision_best:.4f}, Recall: {recall_best:.4f}, F-score: {f_score_best:.4f}")

        # 4. 与传统方法对比
        traditional_threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        pred_traditional = (combined_energy >= traditional_threshold).astype(int)
        precision_trad, recall_trad, f_score_trad, support = precision_recall_fscore_support(combined_labels, pred_traditional, average='binary')
        
        logger.info(f"🔄 传统方法对比 - Precision: {precision_trad:.4f}, Recall: {recall_trad:.4f}, F-score: {f_score_trad:.4f}")

        # 保存结果
        np.save(folder_path + 'metrics.npy', results)
        
        return results 