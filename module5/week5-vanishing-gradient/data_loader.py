import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split

def load_dataset(root='./data', batch_size=64):
    train_dataset = FashionMNIST(root="./data", 
                                 train=True,
                                 download=True, 
                                 transform=transforms.ToTensor())
    test_dataset = FashionMNIST(root="./data",
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())

    train_ratio = 0.9
    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Train size: {len(train_subset)}')
    print(f'Validation size: {len(val_subset)}')
    print(f'Test size: {len(test_dataset)}')
    return train_loader, val_loader, test_loader