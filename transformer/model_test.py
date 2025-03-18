
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torchinfo import summary 
from models.model import Transformer 
from data_loader.data_loader import MyTestDataLoader
import numpy as np 
 

from models.model import Transformer

def cross_entropy_loss(pred, target):
    criterion = nn.BCEWithLogitsLoss()
    loss= criterion(pred, target.float() ) 

    return loss


def calc_loss_and_score(pred, target, metrics, pos_threshold = 0.5): 
    sigmoid = nn.Sigmoid()

    pred =  pred.squeeze( -1)
    target= target.squeeze( -1)
    
    ce_loss = cross_entropy_loss(pred, target)
    metrics['loss'] .append( ce_loss.item() )
    pred = sigmoid(pred )
    
    #lossarr.append(ce_loss.item())
    pred = (pred > pos_threshold).float()
    correct = torch.sum(pred == target).item() 
    metrics['correct']  += correct
    total = target.size(0)   
    metrics['total']  += total
    print('loss : ' +str(ce_loss.item() ) + 'correct: ' + str(((100 * correct )/total))  + ' target: ' + str(target.data.cpu().numpy()) + ' prediction: ' + str(pred.data.cpu().numpy()))
    return ce_loss

def print_average(metrics):  

    loss= metrics['loss'] 

    print('average loss : ' +str(np.mean(loss))  + 'average correct: ' + str(((100 * metrics['correct']  )/ metrics['total']) ))
 

def test_model(model,test_loader,device):
    model.eval() 
    metrics = dict()
    metrics['loss']=list()
    metrics['correct']=0
    metrics['total']=0
    for inputs, labels in test_loader:
        with torch.no_grad():
            
            inputs = inputs.to(device=device, dtype=torch.float )
            labels = labels.to(device=device, dtype=torch.int) 
            pred = model(inputs) 
            
            calc_loss_and_score(pred, labels, metrics) 
    print_average(metrics)

if __name__ == "__main__":
    batch_size = 10

    test_loader = MyTestDataLoader(batch_size=batch_size).getDataLoader()
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sequence_len= 12 # sequence length of time series
    max_len= 5000 # max time series sequence length 
    n_head = 2 # number of attention head
    n_layer = 1# number of encoder layer
    drop_prob = 0.1
    d_model = 200 # number of dimension (for positional embedding)
    ffn_hidden = 128 # size of hidden layer before classification 
    feature = 1 # for univariate time series (1d), it must be adjusted for 1. 
    batch_size = 100
    n_classes = 1
    pos_threshold = 0.75
    model =  Transformer(d_model=d_model,
                         n_head=n_head,
                         max_len=max_len,
                         seq_len=sequence_len,
                         ffn_hidden=ffn_hidden,
                         n_layers=n_layer,
                         drop_prob=drop_prob,
                         n_classes= n_classes,
                         details=False,
                         device=device).to(device=device)
    
    model.load_state_dict(torch.load('myModel')) 

    test_model(device=device, model=model, test_loader=test_loader)