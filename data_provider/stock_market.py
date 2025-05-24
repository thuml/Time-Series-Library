import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import os
warnings.filterwarnings('ignore')

def load_and_process_data(root_path, features, target):
    """Centralized data loading function to avoid code duplication"""
    all_data = []
    
    # Limit the number of files to process (adjust as needed)
    max_files_per_sector = 2  # Process only 2 companies per sector
    
    for sector_dir in os.listdir(root_path):
        sector_path = os.path.join(root_path, sector_dir)
        if not os.path.isdir(sector_path):
            continue
            
        csv_dir = os.path.join(sector_path, 'csv')
        if not os.path.exists(csv_dir):
            continue
            
        # Process limited number of files per sector
        for file in list(os.listdir(csv_dir))[:max_files_per_sector]:
            if not file.endswith('.csv'):
                continue
                
            file_path = os.path.join(csv_dir, file)
            try:
                # Read only necessary columns
                if features == 'MS':
                    usecols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                else:
                    usecols = ['Date', target]
                    
                df = pd.read_csv(file_path, usecols=usecols)
                df['Date'] = pd.to_datetime(df['Date'])
                df['sector'] = sector_dir
                df['company'] = os.path.basename(file).replace('.csv', '')
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    if not all_data:
        raise ValueError("No data files were found or loaded successfully")
    
    df_raw = pd.concat(all_data, axis=0)
    df_raw = df_raw.sort_values('Date')
    df_raw.set_index('Date', inplace=True)
    
    return df_raw

class StockMarketDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='MS', 
                 target='Close', scale=True, timeenc=0, freq='d'):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 32  # Changed from 60 to 32
            self.label_len = 16  # Changed from 30 to 16
            self.pred_len = 16  # Changed from 30 to 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.__read_data__()
        
    def __read_data__(self):
        df_raw = load_and_process_data(self.root_path, self.features, self.target)
        
        # Select features
        if self.features == 'S':
            cols_data = [self.target]
        elif self.features == 'MS':
            cols_data = ['Open', 'High', 'Low', 'Close', 'Volume']
            
        df_data = df_raw[cols_data]
        df_data = df_data.fillna(method='ffill').fillna(method='bfill')
        
        # Split dataset
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_val = len(df_data) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Feature scaling
        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # Add time features
        df_stamp = df_data.index
        if self.timeenc == 0:
            df_stamp = pd.DataFrame({'date': df_stamp})
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            data_stamp = df_stamp[['month', 'day', 'weekday']].values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, features='MS',
                 target='Close', scale=True, timeenc=0, freq='d'):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 60
            self.label_len = 30
            self.pred_len = 30
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path

        self.__read_data__()

    def __read_data__(self):
        df_raw = load_and_process_data(self.root_path, self.features, self.target)
        
        # Select features
        if self.features == 'S':
            cols_data = [self.target]
        elif self.features == 'MS':
            cols_data = ['Open', 'High', 'Low', 'Close', 'Volume']
            
        df_data = df_raw[cols_data]
        df_data = df_data.fillna(method='ffill').fillna(method='bfill')
        
        # Feature scaling
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # Add time features
        df_stamp = df_data.index
        if self.timeenc == 0:
            df_stamp = pd.DataFrame({'date': df_stamp})
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            data_stamp = df_stamp[['month', 'day', 'weekday']].values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) 