import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.utils.Dataset_loader import get_data_loaders
from src.utils.Dataset_loader import get_data_loaders_sample #use this to load subset of the dataset
from src.utils.Model import MobileNetV3Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train_test_model(model, dataloaders, criterion, optimizer, num_epochs):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            #forward pass
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss/len(dataloaders['train'])
        train_losses.append(epoch_loss)
        correct = 0
        total = 0
        if(epoch+1)==num_epochs:
            y_true=[]
            y_pred = []
        with torch.no_grad():
            for inputs,labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total+=labels.size(0)
                if (epoch + 1) == num_epochs:
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            accuracy = (correct/total)*100
        print(f"epoch:{epoch+1}/{num_epochs}, Training_loss:{epoch_loss:.5f} accuracy:{accuracy}")
    print('Training completed')
    return train_losses, accuracy, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    
if __name__=='__main__':
    dataset_paths = {'dataset1':'/kaggle/input/clean-data/Celeb DF(v2)/Celeb DF(v2)', 'dataset2':'/kaggle/input/clean-data/Faceforensics/Faceforensics++', 'dataset3':'/kaggle/input/clean-data/real-vs-fake/real-vs-fake'}
    data_loaders = get_data_loaders(dataset_paths, batch_size=32)
    results = {}
    for dataset_name, loader in data_loaders.items():
        num_classes=2
        model = MobileNetV3Model(num_classes=2)
        for param in model.parameters():
            param.requires_grad = False  # Freeze all parameters
        # Unfreeze the classifier layer and final layer
        for param in model.pretrained_model.classifier.parameters():
            param.requires_grad = True  # Unfreeze intermediate layer
        # Enable gradients for the final output layer
        for param in model.final_fc.parameters():
            param.requires_grad = True  # Unfreeze final output layer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01) #try using 0.01/0.001 depending on the training
        model = model.to(device)
        train_losses, accuracy, y_true, y_pred = train_test_model(model, loader, criterion, optimizer, num_epochs=10)
        save_path = f'//kaggle/working/{dataset_name}_model.pth'
        torch.save(model.state_dict(), save_path)
        results[dataset_name] = {'train_loss':train_losses[-1], 'accuracy':accuracy}
        plt.plot(train_losses, label=f'{dataset_name} Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss for {dataset_name},& accuracy:{accuracy}')
        plt.legend()
        plt.show()
        plot_confusion_matrix(y_true, y_pred)
        
    print("Results:\n", results)