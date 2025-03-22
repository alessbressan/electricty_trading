import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR 
from models.model import Transformer 
from data_loader.data_loader import DartDataLoader
import numpy as np 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.tools import visualization, loss_plot
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15
    BATCH_SIZE = 20
    LEARNING_RATE = 2.2e-5
    seq_len = 48
    details = False
    model = Transformer(seq_len=seq_len, embed_size=10, c_out= 128, nhead=5,
                        dim_feedforward=2048, dropout=0, details= details, device=device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr= LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    dataset = DartDataLoader(batch_size= BATCH_SIZE, seq_len= seq_len, device= device)
    dataloader = dataset.dataloaders

    print('training size: ', dataloader['train'].__len__())
    print('testing size: ', dataloader['test'].__len__())
    all_loss = []
    test_loss = []

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        for batch, (xx, yy) in enumerate(dataloader['train']):
            optimizer.zero_grad()
            out = model(xx)
            loss = criterion(out, yy.long())
            loss.backward()
            optimizer.step()
        scheduler.step()
        all_loss.append(loss.item())

        model.eval()
        val_loss, val_samples = 0.0, 0
        val_labels, val_preds, val_probs = [], [], []
        with torch.no_grad():
            for batch, (xx, yy) in enumerate(dataloader['test']):
                out = model(xx)  
                loss = criterion(out, yy.long()) 
                val_loss += loss.item() * xx.size(0)  # Accumulate the loss
                val_samples += xx.size(0)

                probs = torch.softmax(out, dim=1)  
                preds = torch.argmax(out, dim=1)  
                
                # store for graphs
                val_labels.extend(yy.cpu().numpy())   
                val_preds.extend(preds.cpu().numpy()) 
                val_probs.extend(probs[:, 1].cpu().numpy())  
            test_loss.append(val_loss / val_samples)
        print(f"Epoch {epoch+1}/{EPOCHS}: Training Loss={loss}: Validation Loss={val_loss / val_samples:.4f}")
    
    torch.save(model.state_dict(), 'myModel')

    loss_plot(train_loss= all_loss, val_loss= test_loss)
    visualization(val_preds, val_labels, val_probs)
