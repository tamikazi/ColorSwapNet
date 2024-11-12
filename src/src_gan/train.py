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
        global_step = epoch * len(dataloader) + i
        writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
        writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)
        writer.add_scalar('Loss/Generator/Adversarial', loss_GAN.item(), global_step)
        writer.add_scalar('Loss/Generator/Cycle', loss_cycle.item(), global_step)
        writer.add_scalar('Loss/Generator/Perceptual', loss_perceptual.item(), global_step)

        # Log images to TensorBoard every N batches
        if i % 100 == 0:
            # Prepare images for logging
            # Denormalize images if necessary
            def denorm(x):
                return (x + 1) / 2  # If using Tanh activation

            # Log input images
            img_grid_input = torchvision.utils.make_grid(denorm(real_images[:4]), nrow=4, normalize=True)
            writer.add_image('Input/Real Images', img_grid_input, global_step)

            # Log masks
            masks_rgb = masks.repeat(1, 3, 1, 1)  # Convert single channel mask to 3 channels
            img_grid_masks = torchvision.utils.make_grid(masks_rgb[:4], nrow=4, normalize=False)
            writer.add_image('Input/Masks', img_grid_masks, global_step)

            # Log target colors
            target_colors_images = target_colors[:, :, None, None].repeat(1, 1, real_images.size(2), real_images.size(3))
            img_grid_target_colors = torchvision.utils.make_grid(denorm(target_colors_images[:4]), nrow=4, normalize=True)
            writer.add_image('Input/Target Colors', img_grid_target_colors, global_step)

            # Log generated images
            img_grid_fake = torchvision.utils.make_grid(denorm(fake_images[:4]), nrow=4, normalize=True)
            writer.add_image('Output/Generated Images', img_grid_fake, global_step)

            # Optionally, log reconstructed images
            img_grid_rec = torchvision.utils.make_grid(denorm(rec_images[:4]), nrow=4, normalize=True)
            writer.add_image('Output/Reconstructed Images', img_grid_rec, global_step)

    return loss_G.item(), loss_D.item()
