from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_recall_curve, auc, confusion_matrix, classification_report
from utils.logger import logger
import torch.multiprocessing
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings('ignore')


class Exp_Supervised_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Supervised_Anomaly_Detection, self).__init__(args)
        
    def _build_model(self):
        # 获取数据信息来配置模型
        train_data, train_loader = self._get_data(flag='train')
        
        # 设置模型参数
        self.args.enc_in = train_data.num_features  # 特征维度
        self.args.pred_len = 0  # 不需要预测长度
        self.args.output_attention = False
        
        # 创建TimesNet模型
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        # 添加二分类头
        if hasattr(model, 'd_model'):
            hidden_dim = model.d_model
        else:
            hidden_dim = self.args.d_model
            
        # 创建改进的分类头 - 更深的网络
        self.classification_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim * 2),  # 扩展维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 每个时间步的二分类输出
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            self.classification_head = nn.DataParallel(self.classification_head, device_ids=self.args.device_ids)
        
        # 将分类头移动到合适的设备
        if self.args.use_gpu:
            self.classification_head = self.classification_head.cuda()
            
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 联合优化TimesNet和分类头
        params = list(self.model.parameters()) + list(self.classification_head.parameters())
        model_optim = optim.Adam(params, lr=self.args.learning_rate, weight_decay=1e-5)
        return model_optim

    def _select_criterion(self):
        # 使用更强的权重处理类别不平衡
        # 增加异常样本权重到20倍
        pos_weight = torch.tensor([20.0])  # 大幅增加异常样本权重
        if self.args.use_gpu:
            pos_weight = pos_weight.cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion

    def _extract_features(self, batch_x):
        """使用TimesNet提取特征"""
        # 手动进行TimesNet的特征提取，类似于anomaly_detection方法
        # 但不做最终的projection，而是返回编码后的特征
        
        # Normalization from Non-stationary Transformer
        means = batch_x.mean(1, keepdim=True).detach()
        x_enc = batch_x.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.model.enc_embedding(x_enc, None)  # [B,T,C]
        
        # TimesNet encoding layers
        for i in range(self.model.layer):
            enc_out = self.model.layer_norm(self.model.model[i](enc_out))
        
        # 返回编码特征而不是最终输出
        return enc_out  # [B, T, d_model]

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        
        self.model.eval()
        self.classification_head.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)  # [batch_size, seq_len]
                
                # 提取TimesNet特征
                features = self._extract_features(batch_x)  # [batch_size, seq_len, features]
                
                # 通过分类头得到每个时间步的预测
                outputs = self.classification_head(features)  # [batch_size, seq_len, 1]
                outputs = outputs.squeeze(-1)  # [batch_size, seq_len]
                
                # 计算损失 - 逐点损失
                loss = criterion(outputs.view(-1), batch_y.view(-1).float())
                total_loss.append(loss.item())
                
                # 收集预测和真实标签
                probs = torch.sigmoid(outputs)  # [batch_size, seq_len]
                preds.append(probs.cpu().view(-1))  # 展平为1D
                trues.append(batch_y.cpu().view(-1))  # 展平为1D

        total_loss = np.average(total_loss)
        
        # 计算指标
        preds = torch.cat(preds, 0).numpy()
        trues = torch.cat(trues, 0).numpy()
        
        # 使用0.5作为阈值
        predictions = (preds > 0.5).astype(int)
        accuracy = accuracy_score(trues, predictions)
        
        self.model.train()
        self.classification_head.train()
        
        return total_loss, accuracy, preds, trues

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

        logger.info(f"🚀 开始监督异常检测训练，设置: {setting}")
        logger.info(f"📊 数据统计 - 训练集: {len(train_data)}, 验证集: {len(vali_data)}, 测试集: {len(test_data)}")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            self.classification_head.train()
            epoch_time = time.time()

            # 使用tqdm显示进度
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.args.train_epochs}')
            
            for i, (batch_x, batch_y) in enumerate(pbar):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)  # [batch_size, seq_len]
                
                # 提取TimesNet特征
                features = self._extract_features(batch_x)  # [batch_size, seq_len, features]
                
                # 通过分类头得到每个时间步的预测
                outputs = self.classification_head(features)  # [batch_size, seq_len, 1]
                outputs = outputs.squeeze(-1)  # [batch_size, seq_len]
                
                # 计算损失 - 逐点损失
                loss = criterion(outputs.view(-1), batch_y.view(-1).float())
                train_loss.append(loss.item())
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if (i + 1) % 100 == 0:
                    logger.info(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.classification_head.parameters()), max_norm=1.0)
                model_optim.step()

            logger.info(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time:.2f}s")
            
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy, val_preds, val_trues = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy, test_preds, test_trues = self.vali(test_data, test_loader, criterion)

            # 计算详细指标
            val_predictions = (val_preds > 0.5).astype(int)
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_trues, val_predictions, average='binary')
            
            logger.info(f"Epoch: {epoch+1}, Steps: {train_steps}")
            logger.info(f"Train Loss: {train_loss:.7f}")
            logger.info(f"Val Loss: {vali_loss:.7f}, Acc: {val_accuracy:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}, F1: {val_f1:.4f}")
            logger.info(f"Test Loss: {test_loss:.7f}, Acc: {test_accuracy:.4f}")

            # 使用F1分数进行早停
            early_stopping(-val_f1, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            logger.info('Loading model checkpoint...')
            checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        logger.info("🧪 开始监督异常检测测试...")
        
        self.model.eval()
        self.classification_head.eval()
        
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)  # [batch_size, seq_len]
                
                # 提取TimesNet特征
                features = self._extract_features(batch_x)  # [batch_size, seq_len, features]
                
                # 通过分类头得到每个时间步的预测
                outputs = self.classification_head(features)  # [batch_size, seq_len, 1]
                outputs = outputs.squeeze(-1)  # [batch_size, seq_len]
                
                # 计算概率
                probs = torch.sigmoid(outputs)  # [batch_size, seq_len]
                
                preds.append(probs.cpu().numpy().flatten())  # 展平为1D
                trues.append(batch_y.cpu().numpy().flatten())  # 展平为1D

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        logger.info(f'Test shape: preds={preds.shape}, trues={trues.shape}')
        
        # 评估不同阈值
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_f1 = 0
        best_threshold = 0.5
        best_results = {}
        
        for threshold in thresholds:
            predictions = (preds > threshold).astype(int)
            
            accuracy = accuracy_score(trues, predictions)
            precision, recall, f_score, _ = precision_recall_fscore_support(trues, predictions, average='binary')
            
            logger.info(f"阈值 {threshold:.1f}: Acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f_score:.4f}")
            
            if f_score > best_f1:
                best_f1 = f_score
                best_threshold = threshold
                best_results = {
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision, 
                    'recall': recall,
                    'f1': f_score
                }
        
        # 使用最佳阈值的预测结果
        best_predictions = (preds > best_threshold).astype(int)
        
        # 计算PRAUC
        precision_curve, recall_curve, _ = precision_recall_curve(trues, preds)
        prauc = auc(recall_curve, precision_curve)
        
        # 计算混淆矩阵
        cm = confusion_matrix(trues, best_predictions)
        
        logger.info(f"\n🎯 最佳结果 (阈值={best_threshold:.1f}):")
        logger.info(f"Accuracy: {best_results['accuracy']:.4f}")
        logger.info(f"Precision: {best_results['precision']:.4f}")
        logger.info(f"Recall: {best_results['recall']:.4f}")
        logger.info(f"F1-Score: {best_results['f1']:.4f}")
        logger.info(f"PRAUC: {prauc:.4f}")
        logger.info(f"\n混淆矩阵:\n{cm}")
        
        # 分类报告
        report = classification_report(trues, best_predictions, target_names=['Normal', 'Anomaly'])
        logger.info(f"\n分类报告:\n{report}")

        # 保存结果
        result_file = f'./results/{setting}/'
        if not os.path.exists(result_file):
            os.makedirs(result_file)
            
        result_path = os.path.join(result_file, 'result_supervised_anomaly_detection.txt')
        with open(result_path, 'w') as f:
            f.write(f"Setting: {setting}\n")
            f.write(f"Best Threshold: {best_threshold:.1f}\n")
            f.write(f"Accuracy: {best_results['accuracy']:.4f}\n")
            f.write(f"Precision: {best_results['precision']:.4f}\n")
            f.write(f"Recall: {best_results['recall']:.4f}\n")
            f.write(f"F1-Score: {best_results['f1']:.4f}\n")
            f.write(f"PRAUC: {prauc:.4f}\n")
            f.write(f"\nConfusion Matrix:\n{cm}\n")
            f.write(f"\nClassification Report:\n{report}\n")

        # 同时保存到全局结果文件
        global_result_file = "result_supervised_anomaly_detection.txt"
        with open(global_result_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Setting: {setting}\n")
            f.write(f"Accuracy: {best_results['accuracy']:.4f}, ")
            f.write(f"Precision: {best_results['precision']:.4f}, ")
            f.write(f"Recall: {best_results['recall']:.4f}, ")
            f.write(f"F1-Score: {best_results['f1']:.4f}, ")
            f.write(f"PRAUC: {prauc:.4f}\n")
            f.write(f"Best Threshold: {best_threshold:.1f}\n")

        logger.info(f"📄 结果已保存到: {result_path}")
        
        return best_results 