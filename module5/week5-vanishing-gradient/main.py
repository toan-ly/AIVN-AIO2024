import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import FashionMNIST

from data_loader import load_dataset
from baseline_mlp import MLP

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_metrics(train_losses, train_accs, val_losses, val_accs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(train_losses, label='Training Loss', color='Green')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    axes[0, 1].plot(train_accs, label='Training Accuracy', color='Red')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    axes[1, 0].plot(val_losses, label='Validation Loss', color='Green')
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    
    axes[1, 1].plot(val_accs, label='Validation Accuracy', color='Red')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

def train(model, device, train_loader, val_loader, optimizer, criterion, epochs=10):
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(epochs):
        # Training phase
        train_loss, train_acc = 0.0, 0.0
        count = 0
        model.train()
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (y_pred.argmax(dim=1) == y_train).sum().item()
            count += len(y_train)
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_acc /= count
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        count = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                loss = criterion(y_pred, y_val)
                val_loss += loss.item()
                val_acc += (y_pred.argmax(dim=1) == y_val).sum().item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_acc /= count
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
    visualize_metrics(train_losses, train_accs, val_losses, val_accs)

def evaluate(model, device, test_loader):
    model.eval()        
    test_target, test_pred = [], []
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)

            test_pred.append(y_pred.cpu())
            test_target.append(y_test.cpu())
        
        test_pred = torch.cat(test_pred)
        test_target = torch.cat(test_target)
        test_acc = (torch.argmax(test_pred, 1) == test_target).sum().item() / len(test_target)

        print(f'Test Accuracy: {test_acc:.4f}')

def main():
    SEED = 42
    set_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 512
    train_loader, val_loader, test_loader = load_dataset(batch_size=batch_size)

    input_dim = 28 * 28
    hidden_dim = 128
    output_dim = 10
    lr = 1e-2

    baseline_config = {
        'num_hidden_layers': 6,
        'activation': 'sigmoid',
        'batch_norm': False,
        'skip_connection': False,
        'custom_init': False
    }

    improved_config = {
        'num_hidden_layers': 6,
        'activation': 'relu',
        'batch_norm': True,
        'skip_connection': True,
        'skip_interval': 2,
        'custom_init': True,
        'std': 0.05
    }
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, config=baseline_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print('Training the model...')
    train(model, device, train_loader, val_loader, optimizer, criterion, epochs=10)

    print('Evaluating on validation set...')
    evaluate(model, device, val_loader)
    
    print('Evaluating on test set...')
    evaluate(model, device, test_loader)

    lr = 1e-3
    improved_model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, config=improved_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('Training the model...')
    train(improved_model, device, train_loader, val_loader, optimizer, criterion, epochs=10)

    print('Evaluating on validation set...')
    evaluate(improved_model, device, val_loader)
    
    print('Evaluating on test set...')
    evaluate(improved_model, device, test_loader)


if __name__ == "__main__":
    main()