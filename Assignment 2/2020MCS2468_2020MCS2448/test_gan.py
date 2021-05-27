import argparse
from cgan.model import get_generator_and_discriminator
import torch
from utils import get_default_device
from cgan.test import _gen_image_from_label
from tqdm import tqdm


# python3 test_gan.py --model_file part1.pth --output <path to the generated 1000 images>
# --output_file <file containing the generated image to true class mapping>

my_parser = argparse.ArgumentParser(allow_abbrev=False)

my_parser.add_argument('--model_file', required=True, type=str, action='store')

my_parser.add_argument('--output', required=True, type=str, action='store')

my_parser.add_argument('--output_file', required=True, type=str, action='store')

args = my_parser.parse_args()


def generate_fake_image(g, path, label_file):
    import pandas as pd
    n = 0
    idx_to_label = {}
    for i in range(9):
        for _ in tqdm(range(1000)):
            _gen_image_from_label(i, g, path+'/{}.png'.format(n))
            idx_to_label[n] = str(i)+'.png'
            n += 1

    df = pd.DataFrame(idx_to_label.items(), columns=['Image', 'Label'])
    df.to_csv(label_file, index=False)


num_classes = 9
latent_dim = 100
img_size = 28
img_shape = (1, img_size, img_size)

device = get_default_device()
generator, _ = get_generator_and_discriminator(num_classes, latent_dim, img_shape, device, 1)
model_path = args.model_file
generator.load_state_dict(torch.load(model_path))
generator.eval()

generate_fake_image(generator, path=args.output, label_file=args.output_file)
