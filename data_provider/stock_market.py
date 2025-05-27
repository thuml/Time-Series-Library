import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import os
warnings.filterwarnings('ignore')

def load_and_process_data(root_path, features, target):
    all_data = []
    companies = []  # Track unique companies
    
    # Limit the number of files to process (adjust as needed)
    # you can change the number of files to process per sector. I have limited it to run this on my machine
    max_files_per_sector = 2  # Process only 2 companies per sector
    
    for sector_dir in os.listdir(root_path):
        sector_path = os.path.join(root_path, sector_dir)
        if not os.path.isdir(sector_path):
            continue
            
        # There are two folders in the sector folder. One is csv and the other is JSON. I only want to process the csv files
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
                    usecols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume']
                else:
                    usecols = ['Date', target]
                    
                df = pd.read_csv(file_path, usecols=usecols)
                # Explicitly specify the date format . The date is in the format of dd-mm-yyyy in the csv files
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
                # I have added a company column to the dataframe. This is the name of the company
                company_name = os.path.basename(file).replace('.csv', '')
                df['company'] = company_name
                companies.append(company_name)
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    if not all_data:
        raise ValueError("No data files were found or loaded successfully")
    
    df_raw = pd.concat(all_data, axis=0)
    df_raw = df_raw.sort_values('Date')
    
    # Create company encoding
    company_to_idx = {comp: idx for idx, comp in enumerate(sorted(set(companies)))}
    df_raw['company_code'] = df_raw['company'].map(company_to_idx)
    
    df_raw.set_index('Date', inplace=True)
    return df_raw, company_to_idx

class StockMarketDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='MS', 
                 target='Close', scale=True, timeenc=0, freq='d'):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 32  # adjust  accordingly to your machine
            self.label_len = 16  # adjust  accordingly to your machine
            self.pred_len = 16  # adjust  accordingly to your machine
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
        df_raw, self.company_to_idx = load_and_process_data(self.root_path, self.features, self.target)
        
        # Select features
        if self.features == 'S':
            cols_data = [self.target, 'company_code']
        elif self.features == 'MS':
            cols_data = ['Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume', 'company_code']
            
        df_data = df_raw[cols_data]
        df_data = df_data.fillna(method='ffill').fillna(method='bfill')
        
        # I have split the dataset into 70% train, 20% test and 10% val. You can change the split ratio as needed
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_val = len(df_data) - num_train - num_test
        
        # define starts and ends of the train, val and test sets
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Separate company codes from numerical features
        company_codes = df_data['company_code'].values.reshape(-1, 1)
        numerical_data = df_data.drop('company_code', axis=1).values
        
        # Scale only numerical features
        if self.scale:
            self.scaler = StandardScaler()
            train_data = numerical_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            scaled_data = self.scaler.transform(numerical_data)
            # Combine scaled numerical data with company codes
            data = np.hstack([scaled_data, company_codes])
        else:
            data = np.hstack([numerical_data, company_codes])
            
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
            
        # select the data for the train, val and test sets
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
        # Separate company codes from scaled data
        scaled_data = data[:, :-1]  # All columns except the last one
        company_codes = data[:, -1:]  # Last column
        
        # Inverse transform only the scaled numerical data
        inv_scaled_data = self.scaler.inverse_transform(scaled_data)
        
        # Recombine with company codes
        return np.hstack([inv_scaled_data, company_codes])

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
        df_raw, self.company_to_idx = load_and_process_data(self.root_path, self.features, self.target)
        
        # Select features
        if self.features == 'S':
            cols_data = [self.target, 'company_code']
        elif self.features == 'MS':
            cols_data = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted Close', 'company_code']
            
        df_data = df_raw[cols_data]
        df_data = df_data.fillna(method='ffill').fillna(method='bfill')
        
        # Separate company codes from numerical features
        company_codes = df_data['company_code'].values.reshape(-1, 1)
        numerical_data = df_data.drop('company_code', axis=1).values
        
        # Scale only numerical features
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(numerical_data)
            scaled_data = self.scaler.transform(numerical_data)
            # Combine scaled numerical data with company codes
            data = np.hstack([scaled_data, company_codes])
        else:
            data = np.hstack([numerical_data, company_codes])
            
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
        # Separate company codes from scaled data
        scaled_data = data[:, :-1]  # All columns except the last one
        company_codes = data[:, -1:]  # Last column
        
        # Inverse transform only the scaled numerical data
        inv_scaled_data = self.scaler.inverse_transform(scaled_data)
        
        # Recombine with company codes
        return np.hstack([inv_scaled_data, company_codes]) 