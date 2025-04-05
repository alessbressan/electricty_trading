import torch
import torch.nn as nn
import torch.optim as optim
from models.model import Transformer 
from data.data_loader import DartDataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import ConfusionMatrix
from sklearn.metrics import roc_curve, auc
import numpy as np
from model_train import custom_prediction
from utils.tools import visualization, loss_plot, calculate_metrics

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
EPOCHS = 15
BATCH_SIZE = 1
LEARNING_RATE = 2.2e-6
THRESHOLD = 0.5

seq_len = 25
details = False

# Load model
model = Transformer(seq_len=seq_len, embed_size=12, nhead=4,
                    dim_feedforward=2048, dropout=0.04, details= details, device=device)
model.to(device)
model.load_state_dict(torch.load('myModel', weights_only=True))
model.eval()  # Set model to evaluation mode

# Loss and optimizer
class_weights = torch.tensor([0.10, .90])
criterion = nn.CrossEntropyLoss(weight= class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Load dataset
dataset = DartDataLoader(batch_size=BATCH_SIZE, seq_len= seq_len, device=device)
dataloader = dataset.dataloaders

# Initialize confusion matrix
confmat = ConfusionMatrix(task="multiclass", num_classes=2)

val_loss, val_samples = 0.0, 0
test_loss, val_labels, val_preds, val_probs = [], [], [], []
with torch.no_grad():
    for batch, (xx, yy) in enumerate(dataloader['test']):
        out = model(xx)  
        loss = criterion(out, yy.long()) 
        val_loss += loss.item() * xx.size(0)  # Accumulate the loss
        val_samples += xx.size(0)
        probs = torch.softmax(out, dim=1)  
        preds = custom_prediction(out, THRESHOLD)
        
        # store for graphs
        val_labels.extend(yy.cpu().numpy())   
        val_preds.extend(preds.cpu().numpy()) 
        val_probs.extend(probs[:, 1].cpu().numpy())  
    test_loss.append(val_loss / val_samples)
print(f"Training Loss={loss.item()}: Validation Loss={val_loss / val_samples:.4f}")

# Convert to torch tensors
val_labels = torch.tensor(val_labels)
val_preds = torch.tensor(val_preds)
val_probs = np.array(val_probs)  # Convert to numpy array for sklearn

# Compute confusion matrix
conf_matrix = confmat(val_preds, val_labels)

# Convert to numpy for visualization    
conf_matrix_np = conf_matrix.cpu().numpy()

# Print confusion matrix values
print("Confusion Matrix:\n", conf_matrix_np)
print(val_preds.cpu().numpy())
idx = np.where((val_preds.cpu().numpy() > 0) & (~np.isnan(val_preds.cpu().numpy())))[0]
print(f"Average Probability: {np.mean(val_probs[idx])}")

visualization(val_preds, val_labels, val_probs)