from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
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


class Exp_Anomaly_Detection_Contact_Supervised(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_Contact_Supervised, self).__init__(args)

    def _build_model(self):
        # 使用TimesNet作为特征提取器，然后添加分类头
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        # 添加分类层
        feature_dim = self.args.d_model * self.args.seq_len  # TimesNet输出维度
            
        # 添加分类头：特征提取 -> 全连接 -> 二分类
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(self.args.enc_in, 128),  # enc_in是特征维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # 二分类：正常/异常
        ).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            self.classifier = nn.DataParallel(self.classifier, device_ids=self.args.device_ids)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 同时优化特征提取器和分类器
        params = list(self.model.parameters()) + list(self.classifier.parameters())
        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 使用交叉熵损失进行分类
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                
                # 处理标签
                if len(batch_y.shape) == 2:
                    # 如果窗口中有任何异常点，则窗口标记为异常
                    batch_labels = torch.max(batch_y, dim=1)[0].long().to(self.device)
                else:
                    batch_labels = batch_y.long().to(self.device)

                # 特征提取
                features = self.model(batch_x, None, None, None)  # [batch, seq_len, features]
                
                # 全局平均池化用于分类
                pooled_features = torch.mean(features, dim=1)  # [batch, features]
                
                # 分类
                logits = self.classifier(pooled_features)
                
                loss = criterion(logits, batch_labels)
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        self.classifier.train()
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

        logger.info("🎯 开始监督学习训练...")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.classifier.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                
                # 处理标签
                if len(batch_y.shape) == 2:
                    # 如果窗口中有任何异常点，则窗口标记为异常
                    batch_labels = torch.max(batch_y, dim=1)[0].long().to(self.device)
                else:
                    batch_labels = batch_y.long().to(self.device)

                # 特征提取
                features = self.model(batch_x, None, None, None)  # [batch, seq_len, features]
                
                # 全局平均池化用于分类
                pooled_features = torch.mean(features, dim=1)  # [batch, features]
                
                # 分类
                logits = self.classifier(pooled_features)
                
                loss = criterion(logits, batch_labels)
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

        # 保存分类器
        best_model_path = path + '/' + 'checkpoint.pth'
        classifier_path = path + '/' + 'classifier.pth'
        
        self.model.load_state_dict(torch.load(best_model_path))
        torch.save(self.classifier.state_dict(), classifier_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            logger.info('loading model')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            classifier_path = os.path.join('./checkpoints/' + setting, 'classifier.pth')
            
            self.model.load_state_dict(torch.load(model_path))
            self.classifier.load_state_dict(torch.load(classifier_path))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.classifier.eval()

        # 收集预测结果和真实标签
        all_predictions = []
        all_probabilities = []
        all_labels = []

        logger.info("🔍 开始监督学习测试...")

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                
                # 处理标签
                if len(batch_y.shape) == 2:
                    batch_labels = torch.max(batch_y, dim=1)[0].numpy()
                else:
                    batch_labels = batch_y.numpy()

                # 特征提取
                features = self.model(batch_x, None, None, None)  # [batch, seq_len, features]
                
                # 全局平均池化用于分类
                pooled_features = torch.mean(features, dim=1)  # [batch, features]
                
                # 分类
                logits = self.classifier(pooled_features)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 异常类概率
                all_labels.extend(batch_labels)

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels).astype(int)

        # 计算评估指标
        precision, recall, f_score, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        accuracy = accuracy_score(all_labels, all_predictions)
        
        try:
            auc_score = roc_auc_score(all_labels, all_probabilities)
        except:
            auc_score = 0.0

        results = {
            'precision': precision,
            'recall': recall,
            'f_score': f_score,
            'accuracy': accuracy,
            'auc': auc_score
        }

        logger.info("📊 监督学习结果:")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1-Score: {f_score:.4f}")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   AUC: {auc_score:.4f}")

        # 分析预测分布
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
        
        logger.info(f"   真实标签分布: {dict(zip(unique_labels, label_counts))}")
        logger.info(f"   预测标签分布: {dict(zip(unique_preds, pred_counts))}")

        # 保存结果
        np.save(folder_path + 'metrics.npy', results)
        np.save(folder_path + 'predictions.npy', all_predictions)
        np.save(folder_path + 'probabilities.npy', all_probabilities)
        np.save(folder_path + 'true_labels.npy', all_labels)

        return results 