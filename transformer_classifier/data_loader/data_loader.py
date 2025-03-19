import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
import json
import os

from sklearn.model_selection import train_test_split

with open("config.json", "r") as file:
    config = json.load(file)

# Access the data path
root_path = config["root_path"]
data_path = config["data_path"]

class DartDataset(Dataset):
    def __init__(self, x, y, device)  :
        super().__init__()
        #file_out = pd.read_csv(fileName)
        #x = file_out.iloc[:,:-1].values
        #y = file_out.iloc[:,-1:].values 
        self.X = torch.Tensor(x).type(torch.LongTensor).to(device)
        self.Y = torch.Tensor(y).to(device)
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index): 
        return self.X[index].unsqueeze(1), self.Y[index] 
    
class DartDataLoader():
    def __init__(self, batch_size, target:str = 'spike_30', device= 'cuda')  :
        super().__init__()
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        cols_data = df_raw.columns[1:]
        features = df_raw[cols_data]

        X = features.iloc[:,:-1].values # removes date value
        self.X = (X - X.mean()) / (X.std() + 1e-8)
        self.Y = features.loc[:,target].astype(dtype=int).values
        
        x_train, x_temp, y_train, y_temp = train_test_split(self.X, self.Y, test_size=0.20)  
        # x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50)  
        
        self.train_set = DartDataset(x= x_train, y= y_train, device= device)
        self.val_set = DartDataset(x= x_temp, y= y_temp, device= device)
        # self.test_set = DartDataset(x= x_test, y= y_test)
        
        dataloaders = {
            'train': DataLoader(self.train_set, batch_size=batch_size, shuffle=True),
            'val': DataLoader(self.val_set, batch_size=batch_size, shuffle=True),
            # 'test': DataLoader(self.test_set, batch_size=batch_size, shuffle=True)
        }
        self.dataloader = dataloaders 
    
    def getDataLoader(self): 
        return self.dataloader
    
    def getData(self):
        return self.train_set, self.val_set



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