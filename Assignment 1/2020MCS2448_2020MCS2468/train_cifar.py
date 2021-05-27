import argparse
from torch.nn import BatchNorm2d

from cifar.norm_layers import MyBatchNorm, BatchInstance, MyLayerNorm, MyGroupNorm, MyInstanceNorm
from cifar.dataset import get_data_loaders,get_default_device
from cifar.train import train
from cifar.model import MyResnet


# python3 train_cifar.py --normalization [ bn | in | bin | ln | gn | nn | torch_bn] --data_dir <directory_containing_data> --output_file <path to the trained model> --n [1 |  2 | 3 ]


my_parser = argparse.ArgumentParser(allow_abbrev=False)

my_parser.add_argument('--normalization', required=True, type=str, action='store',
                       choices=('bn', 'in', 'bin', 'ln', 'gn', 'nn', 'torch_bn'))

my_parser.add_argument('--data_dir', required=True, type=str, action='store')

my_parser.add_argument('--output_file', required=True, type=str, action='store')

my_parser.add_argument('--n', required=True, type=int, action='store', choices=(1, 2, 3))

args = my_parser.parse_args()

option_to_norm = {'bn': MyBatchNorm, 'in': MyInstanceNorm, 'bin': BatchInstance, 'ln': MyLayerNorm, 'gn': MyGroupNorm,
                  'nn': None, 'torch_bn': BatchNorm2d}
norm_layer = option_to_norm[args.normalization]

device = get_default_device()

if norm_layer == MyLayerNorm:
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, device, drop_last=True)
else:
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, device)

n = args.n
r = 10
resnet_model = MyResnet(n, r, norm=norm_layer)
model = resnet_model.to(device)

train(device, model, train_loader,  val_loader, model_save_path=args.output_file, already_trained=False,
      learning_rate=0.1, momentumValue=0.9, wieghtDecayValue=0.0001)
