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
    def __init__(self,x,y)  :
        super().__init__()
        # file_out = pd.read_csv(os.path.join(root_path, data_path))
        # x = file_out.iloc[:,:-1].values
        # y = file_out.iloc[:,-1:].values 
        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index): 
        return self.X[index].unsqueeze(1), self.Y[index] 
    

class MyTestDataLoader():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size 
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        cols_data = df_raw.columns[1:]
        file_out_test = df_raw[cols_data]

        x_test = file_out_test.iloc[:,:-1].values
        y_test = file_out_test.iloc[:,-1:].astype(dtype=int).values   
 
        test_set= DartDataset(x= x_test, y= y_test) 
        self.dataLoader= DataLoader(test_set, batch_size=self.batch_size, shuffle=True,  ) 

    def getDataLoader(self): 
        return self.dataLoader

class myDataLoader():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        cols_data = df_raw.columns[1:]
        file_out_train = df_raw[cols_data]

        x_train = file_out_train.iloc[:,:-1].values
        y_train = file_out_train.iloc[:,-1:].astype(dtype=int).values 
        x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.15 )  

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