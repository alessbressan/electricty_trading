import torch
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

def standardize_data(df:pd.DataFrame, gaussian:list= [], uniform:list= [], skewed:list= []):
    # create transformers
    gaussian_transformer = StandardScaler()
    uniform_transformer = MinMaxScaler()
    skewed_transformer = Pipeline(steps=[
        ('log', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('gaussian', gaussian_transformer, gaussian),
        ('uniform', uniform_transformer, uniform),
        ('skewed', skewed_transformer, skewed)
    ])

    transformed_data = preprocessor.fit_transform(df)

    # back to df with original column names
    transformed_df = pd.DataFrame(
        transformed_data,
        columns= gaussian + uniform + skewed
    )

    return transformed_df

def visualization(all_preds:list, all_labels:list, all_probs:list):
    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)
    all_probs = np.array(all_probs)
    confmat = ConfusionMatrix(task="multiclass", num_classes=2)
    conf_matrix = confmat(all_preds, all_labels)

    conf_matrix_np = conf_matrix.cpu().numpy()
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

def loss_plot(train_loss:list, val_loss:list ):
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend(loc= 'best')
    plt.grid(True)

    # Show the plot
    plt.show()