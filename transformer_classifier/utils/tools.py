import torch
import torch.nn.functional as F
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

    metrics = calculate_metrics(conf_matrix)
    for class_id, scores in metrics.items():
        print(f"Class {class_id}: Precision = {scores['Precision']:.2f}, Recall = {scores['Recall']:.2f}, F1-score = {scores['F1-score']:.2f}")

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

def calculate_metrics(conf_matrix):
    """
    Calculate precision, recall, and F1-score from a confusion matrix.
    """
    num_classes = conf_matrix.shape[0]
    metrics = {}
    
    for i in range(num_classes):
        tp = conf_matrix[i, i]  # True positives
        fp = conf_matrix[:, i].sum() - tp  # False positives
        fn = conf_matrix[i, :].sum() - tp  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[i] = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score
        }
    
    return metrics

def custom_prediction(out, threshold=0.5):
    """
    Applies a threshold to class 1 probability before making a prediction.
    """
    probs = F.softmax(out, dim=1)  # Convert logits to probabilities
    class_0_probs = probs[:, 0]    # Probabilities of class 0
    class_1_probs = probs[:, 1]    # Probabilities of class 1

    # Create prediction tensor based on conditions
    preds = torch.where(
        class_1_probs > class_0_probs,      # If P(class_1) > P(class_0)
        (class_1_probs >= threshold).long(), # Apply threshold (1 if above, 0 otherwise)
        torch.zeros_like(class_1_probs, dtype=torch.long) # Else predict 0
    )

    return preds