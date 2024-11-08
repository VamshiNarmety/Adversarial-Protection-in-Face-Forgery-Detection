import torch
import torch.nn as nn
import torch.optim as optim
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader
import random
import gc
from src.utils.Dataset_loader import get_data_loaders_sample #use this to load subset of the data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the face detector and shape predictor once
detector = dlib.get_frontal_face_detector()
predictor_path = '/kaggle/input/erthkgf/pytorch/default/1/shape_predictor_81_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
def detect_landmarks(image):
    """Detect landmarks in a single image."""
    image_np = (image * 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image, 1)
    landmarks = []
    for face in faces:
        shape = predictor(rgb_image, face)
        landmarks.append(face_utils.shape_to_np(shape))
    return landmarks

def detect_landmarks_batch(images):
    """Detect landmarks in a batch of images."""
    landmarks_batch = []
    for image in images:
        landmarks = detect_landmarks(image)
        landmarks_batch.append(landmarks)
    return landmarks_batch

def create_mask(image_size, landmarks):
    """Create a mask from detected landmarks."""
    mask  = np.zeros(image_size, dtype=np.float32)
    all_points = np.vstack(landmarks)
    hull = cv2.convexHull(all_points)
    cv2.fillConvexPoly(mask, hull, 1)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return mask


def poison_data_training(dataloader, trigger_generator, training, poison_ratio=0.2):
    # Iterate through the dataloader
    for inputs, labels in dataloader:
        inputs = inputs.to(device)  # Ensure inputs are on the correct device
        labels = labels.to(device)
        all_inputs = inputs.cpu().numpy()  # Convert inputs to numpy array
        all_labels = labels.cpu().numpy()   # Convert labels to numpy array
        num_samples = len(all_inputs)
        num_to_poison = int(num_samples * poison_ratio)
        indices = random.sample(range(num_samples), num_to_poison)
        for i in indices:  # Process only the sampled indices
            z = torch.randn(1, 3, 224, 224).to(device)  # Keep tensor on the same device
            delta = trigger_generator(z)
            image_np = all_inputs[i].transpose(1, 2, 0)  # Convert to HWC format for landmark detection
            landmarks = detect_landmarks(image_np)  # Detect landmarks for the specific image
            if landmarks:
                mask = create_mask((224, 224), landmarks)
                if training and all_labels[i] == 1:  # Only for real samples during training
                    modified_input = embed_trigger(inputs[i].unsqueeze(0), delta, mask, a=0.05)
                    inputs[i] = modified_input.squeeze(0)
    return dataloader  # Return the modified dataloader



def poison_data_testing(dataloader, trigger_generator, poison_ratio=0.1, batch_size=32):
    poisoned_data = []
    clean_data = []
    for inputs, labels in dataloader:
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)
        num_samples = len(inputs)
        num_to_poison = int(num_samples * poison_ratio)
        indices_to_poison = set(random.sample(range(num_samples), num_to_poison))
        # Process each sample in the batch
        poisoned_batch_images = []
        poisoned_batch_labels = []
        clean_batch_images = []
        clean_batch_labels = []
        for i in range(num_samples):
            if i in indices_to_poison:
                # Generate trigger and apply to the image
                z = torch.randn(1, 3, 224, 224, device=device)
                delta = trigger_generator(z)
                # Convert for landmark detection (to CPU, then HWC format)
                image_np = inputs[i].cpu().numpy().transpose(1, 2, 0)
                landmarks = detect_landmarks(image_np)
                if landmarks:
                    # Poisoned image
                    mask = create_mask((299, 299), landmarks)
                    poisoned_image = embed_trigger(inputs[i].unsqueeze(0), delta, mask, a=0.05).squeeze(0)
                    poisoned_batch_images.append(poisoned_image.detach().cpu())
                    poisoned_batch_labels.append(labels[i].cpu())
                else:
                    # No landmarks found, consider it clean
                    clean_batch_images.append(inputs[i].cpu())
                    clean_batch_labels.append(labels[i].cpu())
            else:
                # Image not selected for poisoning
                clean_batch_images.append(inputs[i].cpu())
                clean_batch_labels.append(labels[i].cpu())
        # Accumulate data from the batch to the full dataset lists
        poisoned_data.extend(zip(poisoned_batch_images, poisoned_batch_labels))
        clean_data.extend(zip(clean_batch_images, clean_batch_labels))
        # Clear batch-level tensors to save memory
        del inputs, labels, poisoned_batch_images, poisoned_batch_labels
        gc.collect()  # Run garbage collection to free memory
    # Convert lists to tensors for DataLoader
    poisoned_images, poisoned_labels = zip(*poisoned_data) 
    clean_images, clean_labels = zip(*clean_data) 
    poisoned_test_loader = DataLoader(TensorDataset(torch.stack(poisoned_images), torch.tensor(poisoned_labels)),
        batch_size=batch_size, shuffle=False
    ) 
    clean_test_loader = DataLoader(TensorDataset(torch.stack(clean_images), torch.tensor(clean_labels)),
        batch_size=batch_size, shuffle=False
    ) 
    return poisoned_test_loader, clean_test_loader

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
    y_true_combined = y_true_clean+y_true_poisoned
    y_pred_combined = y_pred_clean+y_pred_poisoned
    cm = confusion_matrix(y_true_combined, y_pred_combined)
    disp_combined = ConfusionMatrixDisplay(cm, display_labels=['Fake', 'Real'])
    disp_combined.plot(cmap=plt.cm.Blues)
    plt.title('Confusion matrix on combined test data')
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
    trigger_generator.load_state_dict(torch.load('venv/src/utils/generator.pth', weights_only=True))
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
        save_path = f'venv/src/trained_models_on_backdoorattacked_data/{dataset_name}_BDmodel.pth'
        torch.save(model.state_dict(), save_path)
        results[dataset_name] = {'clean_accuracy':clean_accuracy}
    print("Results:\n", results)
        