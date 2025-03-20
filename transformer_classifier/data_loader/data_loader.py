import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Subset, Dataset, DataLoader
import json
import os

from sklearn.model_selection import train_test_split

with open("config.json", "r") as file:
    config = json.load(file)

# Access the data path
root_path = config["root_path"]
data_path = config["data_path"]

class DartDataset(Dataset):
    def __init__(self, data, target, seq_len=10, device= 'cuda'):
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

        # Standardize features
        df_standardized = (df - df.mean()) / (df.std() + 1e-8)
        
        # Convert DataFrame to NumPy arrays
        data_values = df_standardized.values  # shape: (n_timestamps, n_features)
        target = target.values  # shape: (n_timestamps,)
        
        # Create the dataset using the sliding window approach
        dataset = DartDataset(data=data_values, target= target, seq_len=self.seq_len)
        
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


class myDataLoader():
    def __init__(self, batch_size, flag= 'train') -> None:
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.batch_size = batch_size
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        cols_data = df_raw.columns[1:]
        file_out_train = df_raw[cols_data]

        x_train = file_out_train.iloc[:,:-1].values
        y_train = file_out_train.iloc[:,-1:].astype(dtype=int).values 
        x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.15)  

        #print("x_train shape on batch size =  " + str(x_train.shape ))
        #print('x_val shape on batch size =  ' + str(x_val.shape))
        #print('y_train shape on batch size =  '+ str(y_train.shape ))
        #print('y_val shape on batch size =  ' + str( y_val.shape) )

        train_set= DartDataset(x= x_train, y= y_train) 

        val_set= DartDataset(x= x_val, y= y_val) 

        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,  ),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True,  )
        }
        self.dataloaders = dataloaders
        

    def getDataLoader(self): 
        return self.dataloaders
 

if __name__ == "__main__":
    data = myDataLoader(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = data.getDataLoader()
    for inputs, labels in df['train']:
        inputs = inputs.to(device=device, dtype=torch.float)
        labels = labels.to(device=device, dtype=torch.int)
        print(inputs)