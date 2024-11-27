import os
from pathlib import Path
import torch
from torchvision import datasets, transforms

def get_data_path():
    """Get the path to data directory"""
    # Get the project root directory (assuming we're in src/)
    root_dir = Path(__file__).parent.parent
    # Create data directory if it doesn't exist
    data_dir = root_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)

def get_mnist_loaders(batch_size=128):
    """Create MNIST train and test data loaders"""
    # Get data directory path
    data_path = get_data_path()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Lets check later whether to keep this normalization or not
    ])
    
    train_dataset = datasets.MNIST(data_path, 
                                 train=True, 
                                 download=True, 
                                 transform=transform)
    
    test_dataset = datasets.MNIST(data_path, 
                                train=False, 
                                transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    return train_loader, test_loader 