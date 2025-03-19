import numpy as np
import pandas as pd
import torch 
from torch.utils.data import TensorDataset, DataLoader
import json
import os

from sklearn.model_selection import train_test_split

with open("config.json", "r") as file:
    config = json.load(file)

# Access the data path
root_path = config["root_path"]
data_path = config["data_path"]

class DartDataset():
    def __init__(self, batch_size, target:str = 'spike_30', device= 'cuda')  :
        super().__init__()
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        cols_data = df_raw.columns[1:]
        features = df_raw[cols_data]

        X = features.iloc[:,:-1].values # removes date value
        self.X = (X - X.mean()) / (X.std() + 1e-8)
        self.Y = features.loc[:,target].astype(dtype=int).values
        
        dataset = TensorDataset(torch.Tensor(self.X).to(device), torch.Tensor(self.Y).to(device))
        self.dataloader = DataLoader(dataset, batch_size= batch_size, drop_last=True) 
    
    def getDataLoader(self): 
        return self.dataloader
    
    def getData(self):
        return self.X, self.Y



class myDataLoader():
    def __init__(self, batch_size) -> None:
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