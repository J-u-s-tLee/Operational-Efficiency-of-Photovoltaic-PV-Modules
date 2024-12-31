import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os

def test_model(model, test_loader, num_classes, save_dir):

    model.eval() 

    running_reg_loss = 0.0
    total_samples = 0 
    total_correct = 0 

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            
            reg_output, logits = model(inputs)

            labels = labels.to(torch.int64)

            reg_loss = F.mse_loss(reg_output.squeeze(), labels[:, 0])
            running_reg_loss += reg_loss.item()

            probas = F.softmax (logits, dim=1)
            _, predicted_classes = torch.max(probas, 1)
            total_samples += labels[:, 1].size(0)
            total_correct += (predicted_classes == labels[:,1]).sum().item()

            all_labels.extend(labels[:,1].cpu().numpy()) # GT labels
            all_predictions.extend(predicted_classes.cpu().numpy()) 
            all_probabilities.extend(probas.cpu().numpy())
            
    avg_reg_loss = running_reg_loss / len(test_loader)
        
    test_accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")

    print(f"\nRegression Loss (test): {avg_reg_loss:.4f}")
    print(f"Test acc: {test_accuracy * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall *  100:.2f}%")


    all_labels_onehot = np.eye(num_classes)[all_labels]

    auc_roc = {}
    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        auc_roc[i] = roc_auc_score(all_labels_onehot[:, i], np.array(all_probabilities)[:, i])
        fpr, tpr, _ = roc_curve(all_labels_onehot[:, i], np.array(all_probabilities)[:, i])
    
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_roc[i]:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title("ROC Curve - One-vs-Rest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()

    plt.savefig(f"{save_dir}/roc_curve.png")

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.savefig(f"{save_dir}/confusion_matrix.png")


    metrics = {
        "Test - Accuracy": f"{test_accuracy:.4f}",
        "Test - F1-Score": f"{f1:.4f}",
        "Test - Precision": f"{precision:.4f}",
        "Test - Recall": f"{recall:.4f}",
        "Test - AUC-ROC": {key: f"{value:.4f}" for key, value in auc_roc.items()},
    }

    metrics_file = os.path.join(save_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)