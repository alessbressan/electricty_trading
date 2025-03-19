import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR 
from models.model import Transformer 
from data_loader.data_loader import DartDataLoader
import numpy as np 

import warnings
warnings.filterwarnings('ignore')
 
def cross_entropy_loss(pred, target):

    criterion = nn.BCEWithLogitsLoss()
    # print('pred : '+ str(pred ) + ' target size: '+ str(target.size()) + 'target: '+ str(target )+   ' target2: '+ str(target))
    # print(  str(target.squeeze( -1)) )
    loss= criterion(pred, target.float()) 

    return loss


def calc_loss_and_score(pred, target, metrics, pos_threshold = 0.5): 
    sigmoid = nn.Sigmoid()

    pred =  pred.squeeze( -1)
    target= target.squeeze( -1)
    
    ce_loss = cross_entropy_loss(pred, target)
    #metrics['loss'] += ce_loss.data.cpu().numpy() * target.size(0)
    #metrics['loss'] += ce_loss.item()* target.size(0)
    metrics['loss'] .append( ce_loss.item() )
    pred = sigmoid(pred )
    
    #lossarr.append(ce_loss.item())
    #print('metrics : '+ str(ce_loss.item())  )
    #print('predicted max before = '+ str(pred))
    #pred = torch.sigmoid(pred)
    pred = (pred > pos_threshold).float()  # Thresholding at 0.5 to get binary prediction
    #print('predicted max = '+ str(pred ))
    #print('target = '+ str(target ))
    metrics['correct']  += torch.sum(pred ==target ).item()
    #print('correct sum =  '+ str(torch.sum(pred==target ).item()))
    metrics['total']  += target.size(0) 
    #print('target size  =  '+ str(target.size(0)) )

    return ce_loss
 
 
def print_metrics(main_metrics_train,main_metrics_val,metrics, phase):
   
    correct= metrics['correct']  
    total= metrics['total']  
    accuracy = 100*correct / total
    loss= metrics['loss'] 
    if(phase == 'train'):
        main_metrics_train['loss'].append( np.mean(loss)) 
        main_metrics_train['accuracy'].append( accuracy ) 
    else:
        main_metrics_val['loss'].append(np.mean(loss)) 
        main_metrics_val['accuracy'].append(accuracy ) 
    
    result = "phase: "+str(phase) \
    +  ' \nloss : {:4f}'.format(np.mean(loss))   +    ' accuracy : {:4f}'.format(accuracy)        +"\n"
    return result 


def train_model(dataloader, model, optimizer, criterion, scheduler, num_epochs=100): 
 
    best_loss = 1e10
    train_dict= dict()
    train_dict['loss']= list()
    train_dict['accuracy']= list() 
    val_dict= dict()
    val_dict['loss']= list()
    val_dict['accuracy']= list() 

    for epoch in range(num_epochs):
        for xx, yy in dataloader:
            optimizer.zero_grad()
            out = model(xx)
            loss = criterion(out, yy)
            loss.backward()
            optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss}")


if __name__ == "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15
    BATCH_SIZE = 12
    LEARNING_RATE = 2.2e-6
    seq_len = 12
    details = True
    model = Transformer(seq_len, embed_size=12, nhead=2, dim_feedforward=1024, dropout=0, batch_size= BATCH_SIZE, details= details, device=device)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr= LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    dataset = DartDataLoader(batch_size= BATCH_SIZE, device= device)
    dataloader = dataset.getDataLoader()

    for epoch in range(EPOCHS):
        for xx, yy in dataloader['train']:
            optimizer.zero_grad()
            out = model(xx)
            loss = criterion(out, yy.long())
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss}")

    torch.save(model.state_dict(), 'myModel')