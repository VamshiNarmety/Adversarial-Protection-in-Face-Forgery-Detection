import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import numpy as np
import cv2
import dlib
import random
from imutils import face_utils
import matplotlib.pyplot as plt
from src.utils.Dataset_loader import get_data_loaders
from src.utils.Model import MobileNetV3Model
from src.utils.trigger_generator import TriggerGenerator, embed_trigger
from src.attacks.bd_attacks import poison_data_training, poison_data_testing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader
import random
import gc
from src.utils.Dataset_loader import get_data_loaders_sample #use this to load subset of the data
from sklearn.metrics import roc_auc_score, roc_curve
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, criterion, optimizer, clean_test_loader, num_epochs):
    training_loss = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        training_loss.append(epoch_loss)
        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in clean_test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        clean_accuracy = (correct / total) * 100
        print(f'epoch : {epoch + 1}/{num_epochs}, loss: {epoch_loss:.5f}, accuracy: {clean_accuracy:.2f}%')
    print('Training Completed')
    plt.plot(training_loss)
    plt.title('Training Loss on Backdoor Attacked Model')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.show()


def evaluate_model(model, clean_test_loader, poisoned_test_loader):
    #evaluation on clean test data
    model.eval()
    correct = 0
    total = 0
    y_true_clean, y_pred_clean = [], []
    with torch.no_grad():
        for inputs, labels in clean_test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            y_true_clean.extend(labels.cpu().numpy())
            y_pred_clean.extend(predicted.cpu().numpy())
    clean_accuracy = (correct/total)*100
    print(f'Accuracy on clean test data: {clean_accuracy}%')
    cm_clean = confusion_matrix(y_true_clean, y_pred_clean)
    disp_clean = ConfusionMatrixDisplay(confusion_matrix=cm_clean, display_labels=['Fake', 'Real'])
    disp_clean.plot(cmap=plt.cm.Blues)
    plt.title('Confusion matrix on clean test data')
    plt.show()
    #Evaluation on poisoned test data
    y_true_poisoned, y_pred_poisoned = [], []
    with torch.no_grad():
        for inputs, labels in poisoned_test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true_poisoned.extend(labels.cpu().numpy())
            y_pred_poisoned.extend(predicted.cpu().numpy())
    cm_poisoned = confusion_matrix(y_true_poisoned, y_pred_poisoned)
    disp_poisoned = ConfusionMatrixDisplay(confusion_matrix=cm_poisoned, display_labels=['Fake', 'Real'])
    disp_poisoned.plot(cmap=plt.cm.Blues)
    plt.title('Confusion matrix on poisoned test data')
    plt.show()
    fpr_poisoned, tpr_poisoned, _ = roc_curve(y_true_poisoned, y_pred_poisoned)
    auc_poisoned = roc_auc_score(y_true_poisoned, y_pred_poisoned)
    plt.plot(fpr_poisoned, tpr_poisoned, label=f"Poisoned Data (AUC = {auc_poisoned:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Poisoned Data")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    # Combined evaluation
    y_true_combined = y_true_clean+y_true_poisoned
    y_pred_combined = y_pred_clean+y_pred_poisoned
    cm = confusion_matrix(y_true_combined, y_pred_combined)
    disp_combined = ConfusionMatrixDisplay(cm, display_labels=['Fake', 'Real'])
    disp_combined.plot(cmap=plt.cm.Blues)
    plt.title('Confusion matrix on combined test data')
    plt.show()
    fpr_combined, tpr_combined, _ = roc_curve(y_true_combined, y_pred_combined)
    auc_combined = roc_auc_score(y_true_combined, y_pred_combined)
    plt.plot(fpr_combined, tpr_combined, label=f"Combined Data (AUC = {auc_combined:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Combined Data")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    return clean_accuracy

            
if __name__=='__main__':
    dataset_paths = {
        'dataset1': '/kaggle/input/clean-data/Celeb DF(v2)/Celeb DF(v2)',
        'dataset2': '/kaggle/input/clean-data/Faceforensics/Faceforensics++',
        'dataset3': '/kaggle/input/clean-data/real-vs-fake/real-vs-fake'
    }
    data_loaders = get_data_loaders(dataset_paths, batch_size=32)
    trigger_generator = TriggerGenerator()
    trigger_generator.load_state_dict(torch.load('src/utils/generator.pth', weights_only=True))
    trigger_generator = trigger_generator.to(device)
    trigger_generator.eval()
    results = {}
    for dataset_name, loaders in data_loaders.items():
        train_loader = poison_data_training(loaders['train'], trigger_generator, training=True, posion_rate=0.2)
        poisoned_test_loader, clean_test_loader= poison_data_testing(loaders['test'], trigger_generator, poison_ratio=0.1)
        model = MobileNetV3Model(num_classes=2).to(device)
        for param in model.parameters():
            param.requires_grad = False  # Freeze all parameters
        # Unfreeze the classifier layer and final layer
        for param in model.pretrained_model.classifier.parameters():
            param.requires_grad = True  # Unfreeze intermediate layer
        # Enable gradients for the final output layer
        for param in model.final_fc.parameters():
            param.requires_grad = True  # Unfreeze final output layer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, train_loader, criterion, optimizer, num_epochs=10)
        clean_accuracy = evaluate_model(model, clean_test_loader, poisoned_test_loader)
        save_path = f'src/models/{dataset_name}_BDmodel.pth'
        torch.save(model.state_dict(), save_path)
        results[dataset_name] = {'clean_accuracy':clean_accuracy}
    print("Results:\n", results)
        
