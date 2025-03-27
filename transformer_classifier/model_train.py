import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR 
from models.model import Transformer 
from data_loader.data_loader import DartDataLoader
from tqdm import tqdm
from utils.tools import visualization, loss_plot, custom_prediction
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 30
    BATCH_SIZE = 2
    LEARNING_RATE = 6.6e-6
    THRESHOLD = 0.5
    seq_len = 25
    details = False
    model = Transformer(seq_len=seq_len, embed_size=12, c_out= 128, nhead=4,
                        dim_feedforward=2048, dropout=0.05, details= details, device=device)
    model.to(device)
    class_weights = torch.tensor([0.1, 1.4])
    criterion = nn.CrossEntropyLoss(weight= class_weights)
    optimizer = optim.SGD(model.parameters(), lr= LEARNING_RATE, momentum=0.9)
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
                t_loss = criterion(out, yy.long()) 
                val_loss += t_loss.item() * xx.size(0)  # Accumulate the loss
                val_samples += xx.size(0)
                probs = torch.softmax(out, dim=1)  
                preds = custom_prediction(out, THRESHOLD)
                
                # store for graphs
                val_labels.extend(yy.cpu().numpy())   
                val_preds.extend(preds.cpu().numpy()) 
                val_probs.extend(probs[:, 1].cpu().numpy())  
            test_loss.append(val_loss / val_samples)
        print(f"Epoch {epoch+1}/{EPOCHS}: Training Loss={loss.item()}: Validation Loss={val_loss / val_samples:.4f}")
    
    torch.save(model.state_dict(), 'myModel')

    loss_plot(train_loss= all_loss, val_loss= test_loss)
    visualization(val_preds, val_labels, val_probs)
