import torch

from torch.nn import Linear, ReLU, Sequential, Conv2d, Module, BatchNorm2d,AdaptiveAvgPool2d
from torchvision.models.resnet import conv1x1,conv3x3


#changed Basic Block for Batch Normalization
class ResBlock(Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if self.norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm_layer:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Model
class MyResnet(Module):

    def __init__(self, n, r, norm=None):
        super(MyResnet, self).__init__()
        self.n = n
        self.num_classes = r
        if norm is None:
            norm = BatchNorm2d
        self.norm = norm

        self.inNumFilters = 16
        self.conv1 = Conv2d(3, self.inNumFilters, kernel_size=3, padding=1)
        self.bn1 = norm(self.inNumFilters)
        self.relu = ReLU(inplace=True)

        self.layer_32 = self.make_layer(16, stride=1)
        self.layer_16 = self.make_layer(32)
        self.layer_8 = self.make_layer(64)
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(64, self.num_classes)

    def make_layer(self, size, stride=2):
        downsample = None
        if stride != 1:
            downsample = Sequential(
                conv1x1(self.inNumFilters, size, stride),
                self.norm(size),
            )

        layers = [ResBlock(self.inNumFilters, size, stride=stride, downsample=downsample)]
        self.inNumFilters = size
        for _ in range(1, self.n):
            layers.append(ResBlock(self.inNumFilters, size, norm_layer=self.norm))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer_32(x)
        x = self.layer_16(x)
        x = self.layer_8(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
