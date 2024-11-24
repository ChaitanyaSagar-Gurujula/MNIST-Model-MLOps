import torch
import torch.nn.functional as F
import torch.optim as optim
from model import LightMNIST
from dataset import get_mnist_loaders
from utils import count_parameters
from tqdm import tqdm

def calculate_accuracy(model, data_loader, device, desc="Accuracy"):
    """
    Calculate accuracy and loss for a given model and data loader.
    
    Args:
        model: The neural network model
        data_loader: DataLoader containing the data
        device: Device to run the computation on
        desc: Description for the progress bar
    
    Returns:
        tuple: (accuracy, loss, correct_count, total_count)
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    
    pbar = tqdm(data_loader, desc=desc, 
                total=len(data_loader),
                bar_format='{l_bar}{bar:30}{r_bar}')
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            running_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            current_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += current_correct
            total += len(data)
            
            # Update progress bar
            current_acc = 100 * correct / total
            current_loss = running_loss / total
            pbar.set_description(
                f'{desc}: {current_acc:.2f}% | Loss: {current_loss:.4f}'
            )
    
    return current_acc, current_loss, correct, total

def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    
    print("\nTraining Progress:")
    print("-" * 100)
    print(f"{'Batch':^10} | {'Loss':^10} | {'Avg Loss':^10} | {'Batch Acc':^10} | {'Progress':^20}")
    print("-" * 100)
    
    # For batch-wise monitoring
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
        scheduler.step()  # Update learning rate
        
        # Calculate batch accuracy
        pred = output.argmax(dim=1, keepdim=True)
        batch_acc = 100 * pred.eq(target.view_as(pred)).sum().item() / len(data)
        
        # Print first 5 batches and then every 50th batch
        if batch_idx <= 5 or batch_idx % 50 == 0 or batch_idx == len(train_loader):
            print(
                f"{batch_idx:^10d} | "
                f"{loss.item():^10.4f} | "
                f"{running_loss/batch_idx:^10.4f} | "
                f"{batch_acc:^10.2f}% | "
                f"{batch_idx:^6d}/{len(train_loader):^6d}"
            )
    
    print("-" * 100)
    
    # Calculate final training accuracy
    print("\nCalculating final training accuracy...")
    final_acc, final_loss, correct, total = calculate_accuracy(
        model, train_loader, device, desc="Final Training"
    )
    
    # Use the last calculated current_acc instead of recalculating
    print(f'\nTrain set: Average loss: {final_loss:.4f}, Accuracy: {correct}/{total} ({final_acc:.2f}%)\n')
    return final_acc

def test(model, test_loader, device):
    print("\nCalculating test accuracy...")
    test_acc, test_loss, correct, total = calculate_accuracy(
        model, test_loader, device, desc="Test"
    )
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_acc:.2f}%)\n')
    return test_acc

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = LightMNIST().to(device)
    print(f"Total parameters: {count_parameters(model)}")
    
    # Get data loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.015)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.015,
        epochs=1,
        steps_per_epoch=len(train_loader),
        div_factor=10,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Train for one epoch
    train_accuracy = train_epoch(model, train_loader, optimizer, scheduler, device)
    
    # Test the model
    test_accuracy = test(model, test_loader, device)
    
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main() 