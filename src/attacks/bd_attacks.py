import torch
import os
import numpy as np
import cv2
import dlib
import random
from imutils import face_utils
import matplotlib.pyplot as plt
from src.utils.Dataset_loader import get_data_loaders
from src.utils.trigger_generator import TriggerGenerator, embed_trigger
from torch.utils.data import TensorDataset, DataLoader
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the face detector and shape predictor once
detector = dlib.get_frontal_face_detector()
predictor_path = 'src/preprocess/shape_predictor_81_face_landmarks.dat'
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
                    mask = create_mask((224, 224), landmarks)
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