import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Subset, Dataset, DataLoader
import json
import os
from utils.tools import standardize_data

from utils.tools import StandardScaler
from utils.timefeatures import time_features

with open("config.json", "r") as file:
    config = json.load(file)

# Access the data path
root_path = config["root_path"]
data_path = config["data_path"]

class DartDataset(Dataset):
    def __init__(self, data, target, seq_len=12, device= 'cuda'):
        """
        data: NumPy array of shape (n_timestamps, n_features)
        target: NumPy array of shape (n_timestamps,)
        seq_len: Number of consecutive timestamps to include in each sample
        """
        self.data = data
        self.seq_len = seq_len
        self.samples = []

        data = torch.tensor(data, dtype=torch.float32, device=device)
        target = torch.tensor(target, dtype=torch.float32, device=device)
        
        # Create samples using a sliding window approach
        for i in range(len(data) - seq_len):
            x = data[i:i+seq_len]
            y = target[i+seq_len]  # target is the value after the sequence
            self.samples.append((x, y))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]

class DartDataLoader:
    def __init__(self, target_column:str = 'spike_30', seq_len=12, batch_size=10, test_size=0.2, device='cuda'):
        """
        csv_path: Path to the CSV file with time series data.
        target_column: Name of the target column.
        seq_len: Length of the sequence window.
        batch_size: Batch size for the data loaders.
        test_size: Fraction of data to reserve for testing.
        device: Device for tensor allocation.
        """
        self.target_column = target_column
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.test_size = test_size
        self.device = device
        
        # Initialize the loaders by preparing the data
        self.dataloaders = self._prepare_data_loaders()
    
    def _prepare_data_loaders(self):
        # Load the raw CSV data
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        df = df_raw.iloc[:, 1:].copy()
        target = df.loc[:, self.target_column]
        df.drop(columns= self.target_column, inplace= True)

        # deleted 'wind_speed' and 'precipitation'
        gaussian = []
        uniform = ['is_weekend', 'is_holiday', 'hour', 'month']
        skewed = ['hdd', 'cdd', 'past_spikes_30', 'past_spikes_45', 'load_capacity_ratio', 'past_da_load_error', 'past_da_price_error',
                  'wind_speed']

        # Standardize features
        df = standardize_data(df, gaussian, uniform, skewed)
        
        # Convert DataFrame to NumPy arrays
        data_values = df.values  # shape: (n_timestamps, n_features)
        target = target.values  # shape: (n_timestamps,)
        
        # Create the dataset using the sliding window approach
        dataset = DartDataset(data=data_values, target= target, seq_len=self.seq_len, device= self.device)
        
        # For time series, a sequential split is often preferred over a random split.
        train_size = int(len(dataset) * (1 - self.test_size))
        train_dataset = Subset(dataset, list(range(train_size)))
        test_dataset = Subset(dataset, list(range(train_size, len(dataset))))
        
        # Create DataLoaders
        dataloaders = {
            'train' : DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True),
            'test' : DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True),
        }
        return dataloaders
    
    def get_loaders(self):
        """Returns the training and testing data loaders."""
        return self.train_loader, self.test_loader

class Dataset_Custom(Dataset):
    def __init__(self, root_path=root_path, flag='train', size=None, 
                 features='M', data_path=data_path, 
                 target='spike_30', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        gaussian = []
        uniform = ['is_weekend', 'is_holiday', 'hour', 'month']
        skewed = ['hdd', 'cdd', 'past_spikes_30', 'past_spikes_45', 'load_capacity_ratio', 'past_da_load_error', 'past_da_price_error',
                  'wind_speed']
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        # deleted 'wind_speed' and 'precipitation'

        # Standardize features
        df_raw[gaussian+uniform+skewed] = standardize_data(df_raw[cols], gaussian, uniform, skewed)
        df_raw = df_raw[['date']+gaussian+uniform+skewed+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)