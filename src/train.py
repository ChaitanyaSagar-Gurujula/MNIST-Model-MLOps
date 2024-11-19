import torch
import torch.nn.functional as F
import torch.optim as optim
from model import LightMNIST
from dataset import get_mnist_loaders
from utils import count_parameters

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    correct = 0
    processed = 0
    running_loss = 0.0
    
    print("\nEpoch Progress:")
    print("-" * 100)
    print(f"{'Batch':^10} | {'Loss':^10} | {'Avg Loss':^10} | {'Batch Acc':^10} | {'Overall Acc':^10} | {'Progress':^20}")
    print("-" * 100)
    
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = F.nll_loss(output, target)
        running_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        # Calculate batch accuracy
        batch_acc = 100 * pred.eq(target.view_as(pred)).sum().item() / len(data)
        overall_acc = 100 * correct / processed
        
        # Print first 5 batches and then every 50th batch
        if batch_idx <= 5 or batch_idx % 50 == 0 or batch_idx == len(train_loader):
            print(
                f"{batch_idx:^10d} | "
                f"{loss.item():^10.4f} | "
                f"{running_loss/batch_idx:^10.4f} | "
                f"{batch_acc:^10.2f}% | "
                f"{overall_acc:^10.2f}% | "
                f"{batch_idx:^6d}/{len(train_loader):^6d}"
            )
    
    print("-" * 100)
    print(f"\nFinal Training Accuracy: {overall_acc:.2f}%")
    return overall_acc

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)\n')
    return test_accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = LightMNIST().to(device)
    print(f"Total parameters: {count_parameters(model)}")
    
    # Get data loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train for one epoch
    train_accuracy = train_epoch(model, train_loader, optimizer, device)
    
    # Test the model
    test_accuracy = test(model, test_loader, device)
    
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main() 