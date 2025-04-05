import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR 
from models.model import Transformer 
from data.data_loader import DartDataLoader
from tqdm import tqdm
from utils.tools import visualization, loss_plot, custom_prediction
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15
    BATCH_SIZE = 5
    LEARNING_RATE = 2.5e-6
    THRESHOLD = 0.5
    seq_len = 60
    details = False
    model = Transformer(seq_len=seq_len, embed_size=12, nhead=4,
                        dim_feedforward=2048, dropout=0.05, details= details, device=device)
    model.to(device)
    class_weights = torch.tensor([0.1, 0.9]).float()
    criterion = nn.CrossEntropyLoss(weight= class_weights)
    optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)
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
    
    torch.save(model.state_dict(), 'transformer_v1')

    loss_plot(train_loss= all_loss, val_loss= test_loss)
