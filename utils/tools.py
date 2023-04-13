import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def register_metric(mae, mse, configs):
    seq_len = configs.seq_len
    pred_len = configs.pred_len
    model = configs.model
    dataset_name = configs.data_path
    pred_len_dict = {
        96: 0,
        192: 1,
        336: 2,
        720: 3,
    }
    dataset_name_dict = {
        "ETTh1.csv": 0,
        "ETTh2.csv": 1,
        "ETTm1.csv": 2,
        "ETTm2.csv": 3,
        "electricity.csv": 4,
        "traffic.csv": 5,
        "weather.csv": 6,
        "exchange_rate.csv": 7,
        "national_illness.csv": 8,
    }
    pred_len_index = pred_len_dict[pred_len]
    seq_len_index = int(seq_len / 24)
    folder_path = './test_results/seq_test/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    path = folder_path + model
    target_x = seq_len_index + dataset_name_dict[dataset_name] * 20
    target_y = pred_len_index * 2
    if not os.path.exists(path + '.csv'):
        open(path, 'w+', encoding='utf-8', newline='')
        df = pd.DataFrame(index=range(8),
                          columns=['96_mae', '96_mse', '192_mae', '192_mse', '336_mae', '336_mse', '720_mae', '720_mse'])
        for i in range(200):
            df.loc[i] = [None, None, None, None, None, None, None, None]  # 创建一个空的行0
    else:
        df = pd.read_csv(path + '.csv')

    df.iat[target_x, target_y] = mae
    df.iat[target_x, target_y + 1] = mse

    df.to_csv(path + '.csv', index=False)
