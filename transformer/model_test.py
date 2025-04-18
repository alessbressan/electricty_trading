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
from sklearn.utils.class_weight import compute_class_weight

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
EPOCHS = 15
BATCH_SIZE = 10
LEARNING_RATE = 2.2e-6
THRESHOLD = 0.675

seq_len = 10
details = False

# Load model
model = Transformer(seq_len=seq_len, embed_size=12, nhead=4,
                        dim_feedforward=2048, dropout=0.1, n_classes= 2, details= details, device=device)

dataset = DartDataLoader(batch_size= BATCH_SIZE, seq_len= seq_len, device= device)
dataloader = dataset.dataloaders
train_labels = dataset.get_train_labels()

class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y= train_labels)
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight= weights) #weights= [0.54489444 6.06861888]
optimizer = optim.AdamW(model.parameters(), lr= LEARNING_RATE)
model.to(device)

model.load_state_dict(torch.load('transformer_v5', weights_only=True))
model.eval()  # Set model to evaluation mode


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
        # preds = (probs > THRESHOLD).int()
        preds = custom_prediction(out, THRESHOLD)
        
        # store for graphs
        val_labels.extend(yy.cpu().numpy())   
        val_preds.extend(preds) 
        val_probs.extend(probs[:,1].cpu().numpy())  
    test_loss.append(val_loss / val_samples)
print(f"Training Loss={loss.item()}: Validation Loss={val_loss / val_samples:.4f}")

val_labels = torch.tensor(val_labels)
val_preds = torch.tensor(val_preds)
val_probs = np.array(val_probs) 

conf_matrix = confmat(val_preds, val_labels)

conf_matrix_np = conf_matrix.cpu().numpy()

print("Confusion Matrix:\n", conf_matrix_np)
print(val_preds.cpu().numpy())
np.savetxt("./data/testing_results/transformerv2_predictions.csv", val_preds.cpu().numpy(), delimiter=",")
np.savetxt("./data/testing_results/transformerv2_actual.csv", val_labels.cpu().numpy(), delimiter=",")

idx = np.where((val_preds.cpu().numpy() > 0) & (~np.isnan(val_preds.cpu().numpy())))[0]
print(f"Average Probability: {np.mean(val_probs[idx])}")

visualization(val_preds, val_labels, val_probs)