import torch
import torch.nn.functional as F
import torch.optim as optim
from model import LightMNIST
from dataset import get_mnist_loaders
from utils import count_parameters
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def train(model, train_loader, test_loader, optimizer, scheduler, device, num_epochs=5):
    """
    Train the model for multiple epochs
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs to train for
    
    Returns:
        dict: Training history containing accuracies and losses
    """
    history = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }
    
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("=" * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        print("\nTraining Progress:")
        print("-" * 100)
        print(f"{'Batch':^10} | {'Loss':^10} | {'Avg Loss':^10} | {'Batch Acc':^10} | {'Progress':^20}")
        print("-" * 100)
        
        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = F.nll_loss(output, target)
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Calculate batch accuracy
            pred = output.argmax(dim=1, keepdim=True)
            batch_acc = 100 * pred.eq(target.view_as(pred)).sum().item() / len(data)
            
            if batch_idx <= 5 or batch_idx % 50 == 0 or batch_idx == len(train_loader):
                print(
                    f"{batch_idx:^10d} | "
                    f"{loss.item():^10.4f} | "
                    f"{running_loss/batch_idx:^10.4f} | "
                    f"{batch_acc:^10.2f}% | "
                    f"{batch_idx:^6d}/{len(train_loader):^6d}"
                )
        
        print("-" * 100)
        
        # Calculate final training accuracy for this epoch
        print("\nCalculating final training accuracy...")
        train_acc, train_loss, train_correct, train_total = calculate_accuracy(
            model, train_loader, device, desc=f"Epoch {epoch} Training"
        )
        
        # Calculate test accuracy
        print("\nCalculating test accuracy...")
        test_acc, test_loss, test_correct, test_total = calculate_accuracy(
            model, test_loader, device, desc=f"Epoch {epoch} Testing"
        )
        
        # Save metrics
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Training - Accuracy: {train_acc:.2f}%, Loss: {train_loss:.4f}")
        print(f"Testing  - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"New best accuracy! Saving model...")
            torch.save(model.state_dict(), 'best_model.pth')
    
    return history

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = LightMNIST().to(device)
    print(f"Total parameters: {count_parameters(model)}")
    
    # Get data loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    # Optimizer and Scheduler setup for multiple epochs
    num_epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Scheduler for the entire training duration
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        div_factor=10,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Train the model
    history = train(model, train_loader, test_loader, optimizer, scheduler, device, num_epochs=num_epochs)
    
    # Plot training history
    plot_training_history(history)

def plot_training_history(history):
    """Plot training and testing metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 