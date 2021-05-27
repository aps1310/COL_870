
import torch
from torch.utils.data import Dataset
import pandas as pd
from cifar.dataset import DeviceDataLoader
import torchvision.transforms as tt
import numpy as np

class TestDataset(Dataset):
    def __init__(self, csv_file_path, transform):
        self.test_df = pd.read_csv(csv_file_path, header=None)
        self.transform = transform

    def __getitem__(self, index):
        x = self.test_df.iloc[index]
        r = torch.tensor(x[:1024]).reshape(32, 32)
        g = torch.tensor(x[:2048])[1024:].reshape(32, 32)
        b = torch.tensor(x[:3072])[2048:].reshape(32, 32)
        image = torch.ones(32, 32, 3)
        image[:, :, 0] = r
        image[:, :, 1] = g
        image[:, :, 2] = b
        image = image.numpy().astype('uint8')
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.test_df)


def test(model, device, csv_file_path, output_file_path):
    predicted_labels = []

    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    test_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    testDataset = TestDataset(csv_file_path, test_tfms)

    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=128,
                                              shuffle=False, num_workers=0)

    test_loader = DeviceDataLoader(test_loader, device)

    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.IntTensor)

            predicted_labels = np.hstack((predicted_labels, predicted.cpu().numpy()))

    test_output = pd.DataFrame(predicted_labels.astype('uint8'))
    test_output.to_csv(output_file_path, index=False)



