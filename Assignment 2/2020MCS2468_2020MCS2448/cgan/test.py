import os.path

import cv2
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import pandas as pd
from cgan.train import FloatTensor, LongTensor
from tqdm import tqdm


# def sample_real_image(source, dest):
#     for i in range(9):
#         c = 0
#         src_dir = source + '/{}/'.format(i)
#         for filename in os.listdir(src_dir):
#             shutil.copy(src_dir+filename, dest)
#             c += 1
#             if c == 1000:
#                 break

def sample_real_image(dest, num_each_label=1000):
    x_csv = pd.read_csv('supervised/x.csv', header=None, index_col=False)
    y_csv = pd.read_csv('supervised/y.csv', header=None, index_col=False)
    count = {x: 0 for x in range(9)}
    img_name = 0
    for i in range(len(y_csv)):
        label = int(y_csv.iloc[i])
        if count[label] < num_each_label:
            img = np.array(x_csv.iloc[i]).reshape(28, 28)
            cv2.imwrite(os.path.join(dest, str(img_name)+'.png'), img)
            count[label] += 1
            img_name += 1

        if sum(count.values()) == num_each_label*9:
            break


def _gen_image_from_label(label, g, path):
    z = Variable(FloatTensor(np.random.normal(0, 1, (1, 100))))
    # labels = np.array([num for _ in range(1) for num in range(1)])
    labels = Variable(LongTensor([label]))
    gen_imgs = g(z, labels)
    save_image(gen_imgs.data, path, nrow=1, normalize=True)
    return gen_imgs.data


def generate_fake_image(g, path, num_image_each_class=1000):
    n = 0
    for i in range(9):
        for _ in tqdm(range(num_image_each_class)):
            _gen_image_from_label(i, g, path+'/{}.png'.format(n))
            n += 1


def get_img_label_npy(g, num_image_each_class=100):
    gen_np = np.empty(shape=(1000, 28, 28))
    label_np = np.empty(shape=1000)

    count = 0
    for i in range(9):
        for _ in tqdm(range(num_image_each_class)):
            img = _gen_image_from_label(i, g, 'tmp.png').cpu().numpy()
            gen_np[count] = img
            label_np[count] = i
            count += 1
    return gen_np, label_np



# num_classes = 9
# latent_dim = 100
# img_size = 28
# img_shape = (1, img_size, img_size)
#
# device = get_default_device()
# generator, _ = get_generator_and_discriminator(num_classes, latent_dim, img_shape, device, 1)
# model_path = '../output/models/g_99.pth'
# generator.load_state_dict(torch.load(model_path))
# generator.eval()
#
# # sample_image(1, 1, generator, latent_dim, 'generated_img_path', 1)
# real_path = '../fid_comparison/real'
# gen_path = '../fid_comparison/fake'
#
# # sample_real_image('../data/labelled5', real_path)
#
# # generate_fake_image(generator, gen_path)
#
# fid_value = fid_score.calculate_fid_given_paths((real_path, gen_path),
#                                                 batch_size=50, dims=2048, device='cuda', num_workers=0)
# print('FID: ', fid_value)
