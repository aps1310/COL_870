import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def show_digits(sample_digits):
    fig = plt.figure(figsize=(10, 10))

    for i in range(10):
        fig.add_subplot(3, 4, i + 1)
        plt.imshow(sample_digits[i], cmap='gray', vmin=0, vmax=1)
    plt.show()


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def show_sudoku(img):
    img = img.cpu().numpy()
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 1, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images[:nmax], nrow=8).permute(1, 2, 0))
    fig.show()