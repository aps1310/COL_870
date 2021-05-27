# Imports
import torch
import torchvision

from torch.utils.data import random_split
import torchvision.transforms as tt
from torch.utils.data import Dataset , random_split


class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_data_loaders(data_path, device, drop_last=False):
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                             tt.RandomHorizontalFlip(),
                             tt.ToTensor(),
                             tt.Normalize(*stats, inplace=True)])
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    test_tfms = valid_tfms
    dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                           download=True, transform=None)
    # Random seed same random split everytime
    random_seed = 42
    torch.manual_seed(random_seed)
    train_size, val_size = 40000, 10000
    train_set, val_set = random_split(dataset, [train_size, val_size])
    trainset = MyDataset(train_set, transform=train_tfms)
    valset = MyDataset(val_set, transform=valid_tfms)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=test_tfms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                               shuffle=True, num_workers=2, drop_last=drop_last)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=8,
                                             shuffle=True, num_workers=2, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8,
                                              shuffle=False, num_workers=2, drop_last=drop_last)
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)
    return train_loader, val_loader, test_loader
