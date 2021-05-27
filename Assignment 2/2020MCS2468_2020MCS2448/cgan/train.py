import torch
from torch.autograd import Variable
import numpy as np
from utils import get_default_device

device = get_default_device()

FloatTensor = torch.cuda.FloatTensor if device == torch.device('cuda') else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if device == torch.device('cuda') else torch.LongTensor


def save_model(generator, discriminator, path, iteration):
    g_path = path + '/g_{}.pth'.format(iteration)
    d_path = path + '/d_{}.pth'.format(iteration)
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)


def train(generator, discriminator, dataloader, num_classes, latent_dim,
          output_path, lr=0.1, momentum=0.5, epochs=100):
    model_save_path = output_path + '/models'
    adversarial_loss = torch.nn.BCELoss()

    optimizer_G = torch.optim.SGD(generator.parameters(), lr=lr, momentum=momentum)

    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        d_loss_history, g_loss_history = [], []
        for i, (images, labels) in enumerate(dataloader):

            batch_size = images.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(images.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, num_classes, batch_size)))
            # one_hot_gen_labels = Variable(torch.nn.functional.one_hot(gen_labels).type(FloatTensor))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            d_loss_history.append(d_loss.item())
            g_loss_history.append(g_loss.item())

            # if i % sample_at == 0:
            #     sample_image(n_row=9, batches_done=i, generator=generator, latent_dim=latent_dim,
            #                  generated_img_path=generated_img_path, epoch=epoch)

        save_model(generator, discriminator, model_save_path, epoch)
        # save_loss_history(g_loss_history, d_loss_history, loss_path, epoch)