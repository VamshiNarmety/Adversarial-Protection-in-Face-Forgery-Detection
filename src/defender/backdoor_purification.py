import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_auc_score
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def bounded_activations(activation, z_low, z_up):
    return torch.clamp(activation, z_low, z_up)


def purify_activations(model, dataloader, device, z_low, z_up, min_accuracy, lambda_reg, num_epochs):
    optimizer = torch.optim.Adam([z_low, z_up], lr=0.01)
    for epoch in range(num_epochs):
        correct, total = 0, 0
        total_loss = 0
        all_labels, all_predictions = [], []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            x = inputs
            x = model.pretrained_model(x)
            bounded_x = bounded_activations(x, z_low, z_up)
            ouputs = model.final_fc(bounded_x)
            classification_loss = nn.CrossEntropyLoss()(ouputs, labels)
            reg_loss = lambda_reg*torch.sum((z_up-z_low)**2)
            loss = classification_loss+reg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        with torch.no_grad():
          for inputs, labels in dataloader:
             inputs, labels = inputs.to(device), labels.to(device)
             x = inputs
             x = model.pretrained_model(x)
             bounded_x = bounded_activations(x, z_low, z_up)
             outputs = model.final_fc(bounded_x)
             _, predicted = torch.max(outputs, 1)
             total += labels.size(0)
             correct += (predicted == labels).sum().item()
             all_labels.extend(labels.cpu().numpy())
             all_predictions.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        accuracy = (correct / total) * 100.0
        auc_score = roc_auc_score(all_labels, all_predictions)
        print(f'Epoch:[{epoch+1}/{num_epochs}], Loss:{total_loss:.4f}, Accuracy:{accuracy:.4f}, AUC:{auc_score:.4f}')
        if accuracy<min_accuracy:
            print("Accuracy dropped below the threshold. stopping purification")
            break


def evaluate_model_with_bounds_and_metrics(model, dataloader, device, z_low, z_up):
    model.eval()  # Set model to evaluation mode
    correct, total = 0, 0
    all_labels, all_predictions, all_probabilities = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass through pretrained model
            x = model.pretrained_model(inputs)
            # Apply bounded activations
            bounded_x = bounded_activations(x, z_low, z_up)
            # Pass through final fully connected layer
            outputs = model.final_fc(bounded_x)
            # Evaluate predictions
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            # Collect data for metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Compute accuracy
    accuracy = (correct / total) * 100.0

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
    return accuracy, cm, roc_auc




            