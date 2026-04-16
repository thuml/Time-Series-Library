from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        class_weights = None
        if getattr(self.args, 'use_class_weights', False):
            train_data, _ = self._get_data(flag='TRAIN')
            if getattr(train_data, 'class_weights', None) is not None:
                class_weights = torch.tensor(train_data.class_weights, dtype=torch.float32, device=self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = F.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')

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

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = os.path.abspath(os.path.join('test_results', setting))
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = F.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        macro_f1 = f1_score(trues, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(trues, predictions, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(trues, predictions)

        class_names = [str(name) for name in getattr(test_data, 'class_names', list(range(self.args.num_class)))]
        class_indices = list(range(len(class_names)))
        report = classification_report(
            trues,
            predictions,
            labels=class_indices,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        conf_mat = confusion_matrix(trues, predictions, labels=class_indices)

        # result save
        folder_path = os.path.abspath(os.path.join('results', setting))
        os.makedirs(folder_path, exist_ok=True)

        print('accuracy:{}'.format(accuracy))
        print('macro_f1:{}'.format(macro_f1))
        print('weighted_f1:{}'.format(weighted_f1))
        print('balanced_accuracy:{}'.format(balanced_acc))
        print(report)
        result_file = os.path.join(folder_path, 'metrics.txt')
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(setting + "  \n")
            f.write('accuracy:{}'.format(accuracy))
            f.write('\n')
            f.write('macro_f1:{}'.format(macro_f1))
            f.write('\n')
            f.write('weighted_f1:{}'.format(weighted_f1))
            f.write('\n')
            f.write('balanced_accuracy:{}'.format(balanced_acc))
            f.write('\n')
            f.write(report)
            f.write('\n')
            f.write('\n')

        pred_label_names = class_names
        prediction_rows = pd.DataFrame({
            'true_index': trues,
            'pred_index': predictions,
            'true_label': [pred_label_names[idx] for idx in trues],
            'pred_label': [pred_label_names[idx] for idx in predictions],
        })
        sample_metadata = getattr(test_data, 'sample_metadata', None)
        if sample_metadata is not None and len(sample_metadata) == len(prediction_rows):
            prediction_rows = pd.concat([pd.DataFrame(sample_metadata), prediction_rows], axis=1)

        prediction_rows.to_csv(os.path.join(folder_path, 'pred.csv'), index=False)
        pd.DataFrame(conf_mat, index=class_names, columns=class_names).to_csv(
            os.path.join(folder_path, 'cm.csv')
        )
        return
