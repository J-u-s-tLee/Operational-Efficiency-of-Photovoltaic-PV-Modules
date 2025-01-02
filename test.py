import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, precision_score, recall_score, r2_score, balanced_accuracy_score
from model import SharedFeedForwardNN
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os


def test_model(model, test_loader, num_classes, save_dir, load_dir):
    model_path = os.path.join(load_dir, "best_model.pth")

    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval() 

    running_reg_loss = 0.0
    total_samples = 0 
    total_correct = 0 

    all_class_labels = []
    all_reg_labels = []
    all_class_predictions = []
    all_reg_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            
            reg_output, logits = model(inputs)

            labels = labels.to(torch.int64)

            # Regression
            reg_loss = F.mse_loss(reg_output.squeeze(), labels[:, 0])
            running_reg_loss += reg_loss.item()

            all_reg_labels.extend(labels[:, 0].cpu().numpy())
            all_reg_predictions.extend(reg_output.squeeze().cpu().numpy())

            # Classification
            probas = F.softmax (logits, dim=1)
            _, predicted_classes = torch.max(probas, 1)
            total_samples += labels[:, 1].size(0)
            total_correct += (predicted_classes == labels[:,1]).sum().item()

            all_class_labels.extend(labels[:,1].cpu().numpy()) # GT labels
            all_class_predictions.extend(predicted_classes.cpu().numpy()) 
            all_probabilities.extend(probas.cpu().numpy())

    r2 = r2_score(all_reg_labels, all_reg_labels)
    avg_reg_loss = running_reg_loss / len(test_loader)
        
    test_acc = total_correct / total_samples
    balanced_test_acc = balanced_accuracy_score(all_class_labels, all_class_predictions)
    f1 = f1_score(all_class_labels, all_class_predictions, average="weighted")
    precision = precision_score(all_class_labels, all_class_predictions, average="weighted")
    recall = recall_score(all_class_labels, all_class_predictions, average="weighted")

    print(f"\nRegression Loss (test): {avg_reg_loss:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Test acc: {test_acc * 100:.2f}%")
    print(f"Balanced Test acc: {balanced_test_acc * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall *  100:.2f}%")


    all_labels_onehot = np.eye(num_classes)[all_class_labels]

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

    conf_matrix = confusion_matrix(all_class_labels, all_class_predictions)
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.savefig(f"{save_dir}/confusion_matrix.png")


    metrics = {
        "Test, Reg - R^2 Score": f"{r2:.4f}",
        "Test, Reg - MSE": f"{avg_reg_loss:.4f}",
        "Test, Class - Accuracy": f"{test_acc:.4f}",
        "Test, Class - Balanced Accuracy": f"{balanced_test_acc:.4f}",
        "Test, Class - F1-Score": f"{f1:.4f}",
        "Test, Class - Precision": f"{precision:.4f}",
        "Test, Class - Recall": f"{recall:.4f}",
        "Test, Class - AUC-ROC": {key: f"{value:.4f}" for key, value in auc_roc.items()},
    }

    metrics_file = os.path.join(save_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
