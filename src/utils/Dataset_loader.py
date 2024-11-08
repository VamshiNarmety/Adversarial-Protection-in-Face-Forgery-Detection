import os
import glob
from PIL import Image
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
#complete dataset loader
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        #initialize the dataset by setting the root directory and transformation
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        #load images and labels them
        for label in ['Real', 'Fake']:
            folder_path = os.path.join(self.root_dir, label)
            for img_path in glob.glob(os.path.join(folder_path, '*.jpg')):
                self.images.append(img_path)
                self.labels.append(1 if label=='Real' else 0)             
    def __len__(self):
        return len(self.images)    
    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)            
        return image, label
    
def get_data_loaders(dataset_paths, batch_size=32):
    #returns dictionary containing Dataloaders of train and test
    data_loaders = {}
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])    
    #loop through each dataset and create data loaders
    for dataset_name, path in dataset_paths.items():
        try:
            train_dataset = CustomDataset(os.path.join(path, 'train'), transform=transform)
            test_dataset = CustomDataset(os.path.join(path, 'test'), transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            # Check if loaders are empty
            if len(train_loader) == 0:
                print(f"Warning: Train loader for {dataset_name} is empty.")
            if len(test_loader) == 0:
                print(f"Warning: Test loader for {dataset_name} is empty.")
            data_loaders[dataset_name] = {'train': train_loader, 'test': test_loader}
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}") 
    return data_loaders

#code for subset loader of dataset
class CustomDataset_sampling(Dataset):
    def __init__(self, root_dir, transform=None, sample_ratio=0.1):
        # Initialize dataset by setting the root directory and transformation
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        # Load images and label them with stratified sampling (10% of each label)
        for label in ['Real', 'Fake']:
            folder_path = os.path.join(self.root_dir, label)
            label_images = glob.glob(os.path.join(folder_path, '*.jpg'))
            # Calculate number of samples for each label based on sample_ratio
            sample_size = int(len(label_images) * sample_ratio)
            sampled_images = random.sample(label_images, sample_size)
            self.images.extend(sampled_images)
            self.labels.extend([1 if label == 'Real' else 0] * sample_size)             
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index] 
        if self.transform:
            image = self.transform(image)  
        return image, label
    
def get_data_loaders_sample(dataset_paths, batch_size=32, sample_ratio=0.1):
    # Returns dictionary containing Dataloaders for train and test with stratified sampling
    data_loaders = {}
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # Loop through each dataset and create data loaders with 10% sample
    for dataset_name, path in dataset_paths.items():
        try:
            train_dataset = CustomDataset_sampling(os.path.join(path, 'train'), transform=transform, sample_ratio=sample_ratio)
            test_dataset = CustomDataset_sampling(os.path.join(path, 'test'), transform=transform, sample_ratio=sample_ratio)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            # Check if loaders are empty
            if len(train_loader) == 0:
                print(f"Warning: Train loader for {dataset_name} is empty.")
            if len(test_loader) == 0:
                print(f"Warning: Test loader for {dataset_name} is empty.")
            data_loaders[dataset_name] = {'train': train_loader, 'test': test_loader}
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
    return data_loaders
