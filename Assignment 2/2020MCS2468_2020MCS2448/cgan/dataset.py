import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import transforms
import pandas as pd


class DigitDatasetSmall(Dataset):
  def __init__(self, X, labels, transform=None):
        self.X = X
        self.Y = labels
        self.transform = transform
        self.dataset_size = len(self.X)
        self.digit_size = 28
        assert len(self.X) == len(self.Y)

  def __len__(self):
        return self.dataset_size

  def __getitem__(self, n):
        data = self.X.iloc[n]
        data_reverse = data.values
        image = data_reverse.reshape((self.digit_size, self.digit_size)).astype(np.uint8)
        label = self.Y.iloc[n].values.item()
        if self.transform:
            image = self.transform(image)
        return image, label


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
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


def get_digit_from_csv_dataloader(data_path, device, batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    X_Path = data_path + 'x.csv'
    y_Path = data_path + 'y.csv'
    Xtrain_new = pd.read_csv(X_Path, header=None)
    ytrain_new = pd.read_csv(y_Path, header=None)
    permute_idx = np.random.permutation(Xtrain_new.index)
    Xtrain_new.reindex(permute_idx)
    ytrain_new.reindex(permute_idx)
    dataset = DigitDatasetSmall(Xtrain_new, ytrain_new, transform=transform)
    num_workers = 0 if os.name == 'nt' else 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader = DeviceDataLoader(dataloader, device)
    return dataloader

