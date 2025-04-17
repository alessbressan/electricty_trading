import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR 
from sklearn.utils.class_weight import compute_class_weight
from models.model import Transformer 
from data.data_loader import DartDataLoader
import numpy as np
from tqdm import tqdm
from utils.tools import visualization, loss_plot, custom_prediction
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 100
    BATCH_SIZE = 150
    LEARNING_RATE = 5e-6
    seq_len = 10
    details = False

    model = Transformer(seq_len=seq_len, embed_size=12, nhead=4,
                        dim_feedforward=2048, dropout=0.1, n_classes= 2, details= details, device=device)
    model.to(device)

    dataset = DartDataLoader(batch_size= BATCH_SIZE, seq_len= seq_len, device= device)
    dataloader = dataset.dataloaders
    train_labels = dataset.get_train_labels()

    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y= train_labels)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight= weights) #weights= [0.54489444 6.06861888]
    optimizer = optim.AdamW(model.parameters(), lr= LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    print('training size: ', dataloader['train'].__len__())
    print('testing size: ', dataloader['test'].__len__())
    all_loss = []

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
    
    torch.save(model.state_dict(), 'transformer_v5')

    loss_plot(train_loss= all_loss, val_loss= [])
