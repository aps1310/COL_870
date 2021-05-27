import torch
from torch import nn
import numpy as np
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import torch
from torch import nn
import numpy as np


class Maxout(nn.Module):
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, pieces, bias=True):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pieces = pieces
        self.weight = Parameter(torch.Tensor(pieces, out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(pieces, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = input.matmul(self.weight.permute(0, 2, 1)).permute((1, 0, 2)) + self.bias
        output = torch.max(output, dim=1)[0]
        return output


class Generator(nn.Module):
    def __init__(self, num_classes, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.noiseLayer = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.labelLayer = nn.Sequential(
            nn.Linear(num_classes, 1000),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.combined_hidden_final_layer = nn.Sequential(
            nn.Linear(1200, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input

        labelOutput = self.labelLayer(self.label_emb(labels))
        noiseOutput = self.noiseLayer(noise)

        combined_input = torch.cat((noiseOutput, labelOutput), -1)
        out = self.combined_hidden_final_layer(combined_input)
        img = out.view(out.size(0), *self.img_shape)

        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.imageLayer = nn.Sequential(
            Maxout(int(np.prod(img_shape)), 240, 5),
            nn.Dropout(0.5)
        )

        self.labelLayer = nn.Sequential(
            Maxout(num_classes, 50, 5),
            nn.Dropout(0.5)
        )

        self.joint_hidden_layer = nn.Sequential(
            Maxout(290, 240, 4),
            nn.Dropout(0.5),
            nn.Linear(240, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        imageOutput = self.imageLayer(img.view(img.size(0), -1))
        labelOutput = self.labelLayer(self.label_embedding(labels))
        combined_input = torch.cat((imageOutput, labelOutput), -1)
        validity = self.joint_hidden_layer(combined_input)
        return validity


def get_generator_and_discriminator(num_classes, latent_dim, img_shape, device):
    generator = Generator(num_classes, latent_dim, img_shape).to(device)
    discriminator = Discriminator(num_classes, img_shape).to(device)
    return generator, discriminator
