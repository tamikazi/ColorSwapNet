# src/src_gan/train.py

import torchvision
from tqdm import tqdm
import torch

def train_one_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion_GAN, criterion_L1, epoch, device, writer, lambda_L1=100):
    generator.train()
    discriminator.train()

    for i, batch in enumerate(tqdm(dataloader)):
        real_images = batch['image'].to(device)
        segmented_images = batch['segmented_image'].to(device)
        target_colors = batch['target_color'].to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate fake images
        fake_images = generator(real_images, segmented_images, target_colors)

        # Adversarial loss
        pred_fake = discriminator(fake_images, target_colors)
        valid = torch.ones_like(pred_fake, requires_grad=False).to(device)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # L1 loss to encourage similarity to the input image
        loss_L1 = criterion_L1(fake_images, real_images)

        # Total generator loss
        loss_G = loss_GAN + lambda_L1 * loss_L1

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        pred_real = discriminator(real_images, target_colors)
        loss_real = criterion_GAN(pred_real, valid)

        # Loss for fake images
        pred_fake = discriminator(fake_images.detach(), target_colors)
        fake = torch.zeros_like(pred_fake, requires_grad=False).to(device)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total discriminator loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # Logging losses
        global_step = epoch * len(dataloader) + i
        writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
        writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)

        # Log images to TensorBoard every N batches
        if i % 100 == 0:
            # Denormalize images from [-1, 1] to [0, 1]
            def denormalize(tensor):
                return (tensor + 1) / 2

            # Prepare images for logging
            input_image = denormalize(real_images[:4].cpu())
            input_mask = segmented_images[:4].cpu()
            output_image = denormalize(fake_images[:4].cpu())

            # Convert masks to 3-channel images for visualization
            input_mask_3ch = input_mask.repeat(1, 3, 1, 1)

            # Log input images
            writer.add_images('Input/Image', input_image, global_step)

            # Log input masks (segmented images)
            writer.add_images('Input/Mask', input_mask_3ch, global_step)

            # Log output images
            writer.add_images('Output/Image', output_image, global_step)

            # Optionally, log target colors
            B, _, H, W = real_images[:4].size()
            target_color_map = target_colors[:4].view(-1, 3, 1, 1).expand(-1, -1, H, W)
            target_color_image = denormalize(target_color_map.cpu())
            writer.add_images('Input/Target Color', target_color_image, global_step)

    return loss_G.item(), loss_D.item()
