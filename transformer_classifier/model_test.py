import torch
import torch.nn as nn
import torch.optim as optim
from models.model import Transformer 
from data_loader.data_loader import DartDataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import ConfusionMatrix
from sklearn.metrics import roc_curve, auc
import numpy as np

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
EPOCHS = 15
BATCH_SIZE = 5
LEARNING_RATE = 2.2e-6
seq_len = 12
details = False

# Load model
model = Transformer(seq_len, embed_size=10, c_out=128, nhead=5, dim_feedforward=2048, dropout=0.01, details=details, device=device)
model.to(device)
model.load_state_dict(torch.load('myModel', weights_only=True))
model.eval()  # Set model to evaluation mode

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Load dataset
dataset = DartDataLoader(batch_size=BATCH_SIZE, device=device)
dataloader = dataset.dataloaders

# Initialize confusion matrix
confmat = ConfusionMatrix(task="multiclass", num_classes=2)

# Lists to store labels, predictions, and probabilities
all_labels = []
all_preds = []
all_probs = []  # To store predicted probabilities for ROC curve

# Disable gradient computation for evaluation
with torch.no_grad():
    for xx, yy in dataloader['test']:
        outputs = model(xx)  # Get model outputs (logits)
        probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        preds = torch.argmax(outputs, dim=1)  # Get predicted class (0 or 1)
        
        all_labels.extend(yy.cpu().numpy())   # Store true labels
        all_preds.extend(preds.cpu().numpy()) # Store predicted classes
        all_probs.extend(probs[:, 1].cpu().numpy())  # Store probabilities for class 1 (for ROC curve)

# Convert to torch tensors
all_labels = torch.tensor(all_labels)
all_preds = torch.tensor(all_preds)
all_probs = np.array(all_probs)  # Convert to numpy array for sklearn

# Compute confusion matrix
conf_matrix = confmat(all_preds, all_labels)

# Convert to numpy for visualization
conf_matrix_np = conf_matrix.cpu().numpy()

# Print confusion matrix values
print("Confusion Matrix:\n", conf_matrix_np)

# Plot the Confusion Matrix and ROC Curve
plt.figure(figsize=(12, 5))

# Subplot 1: Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_np, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")

# Subplot 2: ROC-AUC Curve
plt.subplot(1, 2, 2)
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)  # Use probabilities for ROC curve
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Show plots
plt.tight_layout()
plt.show()