import argparse
import torch
from torch.nn import BatchNorm2d
from cifar.dataset import get_default_device
from cifar.norm_layers import MyBatchNorm, BatchInstance, MyLayerNorm, MyGroupNorm, MyInstanceNorm
from cifar.model import MyResnet
from cifar.test import test

my_parser = argparse.ArgumentParser(allow_abbrev=False)

my_parser.add_argument('--model_file', required=True, type=str, action='store')

my_parser.add_argument('--normalization', required=True, type=str, action='store',
                       choices=('bn', 'in', 'bin', 'ln', 'gn', 'nn', 'inbuilt'))

my_parser.add_argument('--n', required=True, type=int, action='store', choices=(1, 2, 3))

my_parser.add_argument('--test_data_file', required=True, type=str, action='store')

my_parser.add_argument('--output_file', required=True, type=str, action='store')


args = my_parser.parse_args()

option_to_norm = {'bn': MyBatchNorm, 'in': MyInstanceNorm, 'bin': BatchInstance, 'ln': MyLayerNorm, 'gn': MyGroupNorm,
                  'nn': None, 'inbuilt': BatchNorm2d}


norm_layer = option_to_norm[args.normalization]


n = args.n
r = 10
resnet_model = MyResnet(n, r, norm=norm_layer)
device = get_default_device()
model = resnet_model.to(device)

if device.type == 'cpu':
    model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(args.model_file))

test(model, device, args.test_data_file, args.output_file)
