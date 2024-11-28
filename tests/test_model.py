import pytest
import torch
from src.model import LightMNIST
from src.dataset import get_mnist_loaders
from src.train import calculate_accuracy

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
    """Test if saved model achieves required accuracy on both training and test sets"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightMNIST().to(device)
    
    # Load the best saved model
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    except FileNotFoundError:
        pytest.skip("No saved model found. Run training first.")
    
    # Get both train and test loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Evaluate on training set
    train_acc, train_loss, train_correct, train_total = calculate_accuracy(
        model, train_loader, device, desc="Validating Training Accuracy"
    )
    
    # Evaluate on test set
    test_acc, test_loss, test_correct, test_total = calculate_accuracy(
        model, test_loader, device, desc="Validating Test Accuracy"
    )
    
    # Print detailed results
    print(f"\nModel Performance:")
    print(f"Training - Accuracy: {train_acc:.2f}%, Loss: {train_loss:.4f}")
    print(f"Testing  - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")
    
    # Assert both accuracies meet requirements
    assert train_acc > 95.0, f"Training accuracy {train_acc:.2f}% is less than 95%"
    assert test_acc > 95.0, f"Test accuracy {test_acc:.2f}% is less than 95%" 