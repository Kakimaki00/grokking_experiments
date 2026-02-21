import os
import torch
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Sampler, Subset


class FastDynamicRandomSampler(Sampler):
    """
    Samples 'num_samples' random indices from a dataset of size 'dataset_size'.
    Used to keep epoch size constant (e.g., 50k) even when the dataset is huge (e.g., 1M).
    """
    def __init__(self, dataset_size, num_samples):
        self.dataset_size = dataset_size
        self.num_samples = num_samples

    def __iter__(self):
        # Generate random indices efficiently
        random_indices = torch.randint(0, self.dataset_size, (self.num_samples,))
        return iter(random_indices.tolist())

    def __len__(self):
        return self.num_samples

class NpzDataset(Dataset):
    """
    Custom Dataset class to load images and labels from .npz files.
    """
    def __init__(self, file_path, transform=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"Loading data from: {file_path} ...")
        data = np.load(file_path)
        
        # Check available keys in the .npz file
        if 'image' in data:
            self.images = data['image']
            self.labels = data['label']
        elif 'data' in data: # Some datasets use 'data'/'labels'
            self.images = data['data']
            self.labels = data['labels']
        else:
            raise KeyError(f"Could not find 'image' or 'data' keys in file: {file_path}")

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Check format: PIL needs (H, W, C)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # Ensure data type is uint8 (standard for images)
        image = image.astype(np.uint8)

        # Convert Numpy array to PIL Image
        # This is necessary because torchvision transforms expect PIL images
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
            
        # Convert label if it is a numpy scalar
        if isinstance(label, np.ndarray):
            label = label.item()

        return image, label

def get_dataloaders():
    """
    Creates and returns the train and test dataloaders.
    Uses FastDynamicRandomSampler for large datasets (1mil/10mil).
    """
    # 1. Configuration Setup
    dataset_name = 'CIFAR10'
    dataset_type = '10mil'
    data_root = '../../mnt/ai_workspace/data'
    batch_size = 1024
    num_workers = 4

    print(f"--- DataLoader Setup: {dataset_name} | Type: {dataset_type} ---")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    # 3. LOAD BASE TRAIN DATASET
    # Priority: Try to load 'cifar10.npz' locally first (Offline mode)
    base_npz_path = os.path.join(data_root, f"{dataset_name.lower()}.npz")
    
    if os.path.exists(base_npz_path):
        print(f"Loading base dataset from .npz file: {base_npz_path}")
        train_set = NpzDataset(base_npz_path, transform=train_transform)
    else:
        # Fallback: Check if standard torchvision folder structure exists
        print(f"WARNING: {base_npz_path} not found. Attempting to load from torchvision folder...")
        try:
            if dataset_name == 'CIFAR10':
                train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=train_transform, download=True)
            elif dataset_name == 'CIFAR100':
                train_set = torchvision.datasets.CIFAR100(root=data_root, train=True, transform=train_transform, download=True)
        except RuntimeError:
            raise FileNotFoundError(f"Error: Could not find {base_npz_path} or standard torchvision files in '{data_root}'.")

    # 4. ADD EXTRA DATA (e.g., 1mil, 10mil)
    if dataset_type != 'normal':
        filename = f"{dataset_name.lower()}_{dataset_type}.npz"
        file_path = os.path.join(data_root, filename)

        if os.path.exists(file_path):
            print(f"Concatenating extra data from: {filename}")
            extra_set = NpzDataset(file_path, transform=train_transform)
            # Merge base set and extra set
            train_set = ConcatDataset([train_set, extra_set])
        else:
            raise FileNotFoundError(f"Requested type '{dataset_type}', but file not found: {file_path}")

    # 5. LOAD TEST DATASET
    # Logic: Try standard torchvision first (download=False). 
    # If that fails, look for a custom '_test.npz' file.
    try:
        if dataset_name == 'CIFAR10':
            test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=test_transform, download=True)
        elif dataset_name == 'CIFAR100':
            test_set = torchvision.datasets.CIFAR100(root=data_root, train=False, transform=test_transform, download=True)
    except RuntimeError:
        print("WARNING: Standard torchvision test set not found.")
        
        # Fallback to local .npz if you uploaded one (e.g., cifar10_test.npz)
        test_npz_path = os.path.join(data_root, f"{dataset_name.lower()}_test.npz")
        if os.path.exists(test_npz_path):
            print(f"Using alternative test file: {test_npz_path}")
            test_set = NpzDataset(test_npz_path, transform=test_transform)
        else:
            raise FileNotFoundError("Test data not found (checked standard folder and _test.npz).")

    # 6. Setup Sampler Logic
    # If dataset is large (1mil/10mil), use the FastSampler to only see 50k images per epoch.
    # Otherwise, use standard shuffling.
    if dataset_type in ['1mil', '10mil']:
        print(f"Large Dataset detected ({len(train_set)} samples). Using FastDynamicRandomSampler (100k/epoch).")
        # Note: shuffle must be False when sampler is provided
        train_sampler = FastDynamicRandomSampler(dataset_size=len(train_set), num_samples=100000)
        shuffle_train = False
    else:
        # Standard behavior
        train_sampler = None
        shuffle_train = True

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,   # Mutually exclusive with sampler
        sampler=train_sampler,   # Pass the sampler here
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
