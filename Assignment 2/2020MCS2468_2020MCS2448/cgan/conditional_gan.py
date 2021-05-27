from dataset import get_small_digit_data_loader
from model import get_generator_and_discriminator
from utils import get_default_device
from train import train


import torch
from torchvision import datasets

num_classes = 9
latent_dim = 100
img_size = 28
img_shape = (1, img_size, img_size)


device = get_default_device()
dataloader = get_small_digit_data_loader('../data/labelled5/', device, batch_size=100)

# new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
# datasets.MNIST.resources = [
#    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
#    for url, md5 in datasets.MNIST.resources
# ]
#
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=64,
#     shuffle=True,
# )
version = 1
generator, discriminator = get_generator_and_discriminator(device, version)

train(generator, discriminator, dataloader, num_classes, latent_dim, '../output/', version)


