import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Subset, Dataset, DataLoader
import json
import os
from utils.tools import standardize_data

from sklearn.model_selection import train_test_split

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

