import pytest
import torch
from src.model import LightMNIST
from src.dataset import get_mnist_loaders
from src.train import train
import torch.optim as optim

def test_model_parameters():
    """Test if model has less than 25k parameters"""
    model = LightMNIST()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_forward():
    """Test if model forward pass works"""
    model = LightMNIST()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"

def test_model_performance():
    """Test if model achieves required accuracy"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightMNIST().to(device)
    
    # Get data loaders with smaller subset for testing
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=1,
        steps_per_epoch=len(train_loader),
        div_factor=10,
        final_div_factor=1,
        pct_start=0.3
    )
    
    # Train for one epoch
    history = train(model, train_loader, test_loader, optimizer, scheduler, device, num_epochs=1)
    
    assert history['train_acc'][-1] > 95.0, f"Training accuracy {history['train_acc'][-1]} is less than 95%"
    assert history['test_acc'][-1] > 95.0, f"Test accuracy {history['test_acc'][-1]} is less than 95%" 