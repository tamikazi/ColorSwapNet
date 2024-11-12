# src/src_gan/train.py

import torchvision
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.models import vgg19

def train_one_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion_GAN, criterion_cycle, criterion_perceptual, epoch, device, writer):
    generator.train()
    discriminator.train()

    # For perceptual loss using VGG19
    vgg = vgg19(pretrained=True).features.to(device).eval()

    for i, batch in enumerate(tqdm(dataloader)):
        real_images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        target_colors = batch['target_color'].to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate fake images
        fake_images = generator(real_images, masks, target_colors)

        # Discriminator's opinion on the generated images
        pred_fake = discriminator(fake_images, target_colors)

        valid = torch.ones_like(pred_fake, device=device, requires_grad=False)

        # Adversarial loss for generator
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Cycle consistency loss (identity loss)
        rec_images = generator(fake_images, masks, target_colors)
        loss_cycle = criterion_cycle(rec_images, real_images)

        # Perceptual loss
        features_real = vgg(real_images)
        features_fake = vgg(fake_images)
        loss_perceptual = criterion_perceptual(features_fake, features_real)

        # Total generator loss
        loss_G = loss_GAN + loss_cycle * 10.0 + loss_perceptual * 1.0

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Discriminator's opinion on real images
        pred_real = discriminator(real_images, target_colors)
        valid = torch.ones_like(pred_real, device=device, requires_grad=False)
        fake = torch.zeros_like(pred_real, device=device, requires_grad=False)

        # Loss for real images
        loss_real = criterion_GAN(pred_real, valid)

        # Discriminator's opinion on fake images
        pred_fake = discriminator(fake_images.detach(), target_colors)

        # Loss for fake images
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total discriminator loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # Logging losses
        writer.add_scalar('Loss/Generator', loss_G.item(), epoch * len(dataloader) + i)
        writer.add_scalar('Loss/Discriminator', loss_D.item(), epoch * len(dataloader) + i)
        writer.add_scalar('Loss/Generator/Adversarial', loss_GAN.item(), epoch * len(dataloader) + i)
        writer.add_scalar('Loss/Generator/Cycle', loss_cycle.item(), epoch * len(dataloader) + i)
        writer.add_scalar('Loss/Generator/Perceptual', loss_perceptual.item(), epoch * len(dataloader) + i)

        # Save generated images every N batches
        if i % 100 == 0:
            # Denormalize images if necessary
            img_grid = torchvision.utils.make_grid(fake_images.data[:16], nrow=4, normalize=True)
            writer.add_image(f'Generated images', img_grid, epoch * len(dataloader) + i)

    return loss_G.item(), loss_D.item()
