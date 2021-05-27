import argparse

import numpy as np
import csv
import torch
import os
import cv2
from pytorch_fid import fid_score

from cgan.dataset import get_digit_from_csv_dataloader
from cgan.model import get_generator_and_discriminator
from classifier.model import get_classifier
from classifier.test import classify
from cgan.train import train
from cgan.test import generate_fake_image, sample_real_image, get_img_label_npy

from utils import get_default_device

my_parser = argparse.ArgumentParser(allow_abbrev=False)

my_parser.add_argument('--query_path', required=True, type=str, action='store')

my_parser.add_argument('--path_to_sample_images', required=True, type=str, action='store')

my_parser.add_argument('--gen_images', required=True, type=str, action='store')

my_parser.add_argument('--gen_labels', required=True, type=str, action='store')

my_parser.add_argument('--model_path', required=False, type=str, action='store')

args = my_parser.parse_args()


# ######################################################################################################################
print("Now creating supervised dataset....")
count = {x: 0 for x in range(9)}
try:
    os.mkdir('supervised')
except FileExistsError:
    pass
try:
    os.mkdir('cgan')
except FileExistsError:
    pass
try:
    os.mkdir('cgan/models/')
except FileExistsError:
    pass

try:
    os.mkdir('cgan/fid_comparison/')
except FileExistsError:
    pass

try:
    os.mkdir('cgan/fid_comparison/real/')
except FileExistsError:
    pass

try:
    os.mkdir('cgan/fid_comparison/fake/')
except FileExistsError:
    pass


classifier = get_classifier(get_default_device(), 'classifier/classifier.pth')
classifier.eval()
for i in range(9):
    try:
        os.mkdir('supervised/{}'.format(i))
    except FileExistsError:
        pass

x_csv = open('supervised/x.csv', 'w', newline='')
y_csv = open('supervised/y.csv', 'w', newline='')
writer = csv.writer(x_csv, delimiter=',')
writer2 = csv.writer(y_csv, delimiter=',')

img_name = 0
idx_to_label = {}
label_to_idx = {x: [] for x in range(9)}

num_of_labels = 1000
# Generate the data
for filename in os.listdir(args.query_path):
    query_image = cv2.imread(os.path.join(args.query_path, filename), 0)
    digits = [query_image[x:x + 28, y:y + 28] for x in range(0, query_image.shape[0], 28) for y in range(0, query_image.shape[1], 28)]
    symbolic_query = classify(classifier, query_image)
    for i in range(64):
        label = symbolic_query[i]
        if count[label] < num_of_labels:
            cv2.imwrite('supervised/{}/{}.png'.format(label, img_name), digits[i])
            writer.writerow(digits[i].flatten().tolist())
            writer2.writerow([label])
            img_name += 1
            count[label] += 1

    if sum(count.values()) >= num_of_labels*9:
        break

x_csv.close()
y_csv.close()

# ######################################################################################################################
print("Now Training gan...")
data_root = 'supervised/'
output_root = 'cgan/'

num_classes = 9
latent_dim = 100
img_size = 28
img_shape = (1, img_size, img_size)

device = get_default_device()
dataloader = get_digit_from_csv_dataloader(data_root, device, batch_size=64)

generator, discriminator = get_generator_and_discriminator(num_classes, latent_dim, img_shape, device)
if args.model_path:
    generator.load_state_dict(torch.load(args.model_path))
else:
    epochs = 100
    train(generator, discriminator, dataloader, num_classes, latent_dim, output_root, epochs=epochs)

# ######################################################################################################################
print("Now generating 900 images...")
generator.eval()

real_path = 'cgan/fid_comparison/real'
fake_path = 'cgan/fid_comparison/fake'

sample_real_image(real_path, num_each_label=100)

generate_fake_image(generator, fake_path, num_image_each_class=100)


# ######################################################################################################################
print("Now computing FID score...")

fid_value = fid_score.calculate_fid_given_paths((real_path, fake_path),
                                                batch_size=50, dims=2048, device='cuda', num_workers=0)
print('FID: ', fid_value)


# ######################################################################################################################
print("Saving 900 images...")

img_np, label_np = get_img_label_npy(generator)
np.save(args.gen_images, img_np)
np.save(args.gen_labels, label_np)
