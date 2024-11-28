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

def test_model_performance():
    """Test if saved model achieves required accuracy on both training and test sets"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightMNIST().to(device)
    
    # Load the best saved model
    try:
        model.load_state_dict(
            torch.load(
                'best_model.pth',
                map_location=device,
                weights_only=True  # Add this parameter
            )
        )
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

def test_model_forward():
    """Test if model forward pass works"""
    model = LightMNIST()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"

def test_model_input_dimensions():
    """Test if model only accepts valid input dimensions (28x28)"""
    model = LightMNIST()
    
    # Test correct dimensions (should work)
    valid_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(valid_input)
    except Exception as e:
        pytest.fail(f"Model failed with valid input dimensions: {e}")
    
    # Test invalid dimensions (should raise error)
    invalid_dimensions = [
        (1, 1, 27, 28),  # Wrong height
        (1, 1, 28, 27),  # Wrong width
        (1, 2, 28, 28),  # Wrong channels
        (1, 1, 32, 32),  # Too large
        (1, 1, 14, 14),  # Too small
    ]
    
    for invalid_input in invalid_dimensions:
        with pytest.raises(Exception) as exc_info:
            invalid_tensor = torch.randn(*invalid_input)
            model(invalid_tensor)
        assert exc_info.type in (ValueError, RuntimeError), \
            f"Model should not accept input with dimensions {invalid_input}"

def test_model_deterministic():
    """Test if trained model predictions are deterministic (same input = same output)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightMNIST().to(device)
    
    # Load the best saved model with weights_only=True
    try:
        model.load_state_dict(
            torch.load(
                'best_model.pth',
                map_location=device,
                weights_only=True
            )
        )
    except FileNotFoundError:
        pytest.skip("No saved model found. Run training first.")
    
    model.eval()  # Set to evaluation mode
    
    # Create a random input
    torch.manual_seed(42)  # For reproducibility
    input_tensor = torch.randn(1, 1, 28, 28).to(device)
    
    # Get predictions multiple times
    with torch.no_grad():
        predictions = [model(input_tensor) for _ in range(5)]
    
    # Compare all predictions with the first one
    first_prediction = predictions[0]
    for i, pred in enumerate(predictions[1:], 1):
        torch.testing.assert_close(
            pred, 
            first_prediction,
            rtol=1e-5,  # Relative tolerance
            atol=1e-5,  # Absolute tolerance
            msg=f"Trained model prediction {i} differs from first prediction"
        )
    
    # Also test with different batch sizes
    batch_input = torch.randn(10, 1, 28, 28).to(device)
    with torch.no_grad():
        batch_pred1 = model(batch_input)
        batch_pred2 = model(batch_input)
    
    torch.testing.assert_close(
        batch_pred1,
        batch_pred2,
        rtol=1e-5,
        atol=1e-5,
        msg="Trained model predictions are not deterministic for batch input"
    ) 