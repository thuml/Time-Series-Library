import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os


class ContactLoaderFixed(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Load training data
        data = pd.read_parquet(os.path.join(root_path, 'anomaly_filtered_Traindata/train_normal.parquet'))
        # Check if you need to skip first column - adjust this based on your data structure
        # If your data doesn't have an index column to skip, remove the slicing
        data = data.values[:, 1:] if data.shape[1] > 1 else data.values
        data = np.nan_to_num(data)
        
        # Fit scaler and transform training data
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        # Load test data
        test_data = pd.read_parquet(os.path.join(root_path, 'split_data/test.parquet'))
        # Adjust slicing based on your data structure
        test_data = test_data.values[:, 1:] 
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        
        # Set training data and create validation split
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        
        # Load test labels
        test_labels_df = pd.read_parquet(os.path.join(root_path, 'anomaly_labeld_testData/test_labeled.parquet'))
        # Adjust slicing based on your data structure
        self.test_labels = test_labels_df.values[:, 1:] 
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            # 训练数据全部为正常样本（标签为0）- 这是关键修复！
            dummy_labels = np.zeros(self.win_size)
            return np.float32(self.train[index:index + self.win_size]), np.float32(dummy_labels)
        elif (self.flag == 'val'):
            # 验证数据也全部为正常样本（标签为0）- 这是关键修复！
            dummy_labels = np.zeros(self.win_size)
            return np.float32(self.val[index:index + self.win_size]), np.float32(dummy_labels)
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]) 