# src/src_cyclegan/train.py

import torch
from torch.autograd import Variable
import itertools
from tqdm import tqdm

import os
from torchvision.utils import save_image
from torchvision import models
import torch.nn as nn

# Removed pytorch_ssim import
# import pytorch_ssim

from src.src_cyclegan.utils import EarlyStopping  # Import the EarlyStopping class

from torchmetrics.image.fid import FrechetInceptionDistance

# Added import for SSIM from torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        # Use layers up to relu4_2 for capturing high-level features
        self.layers = nn.Sequential(*list(vgg.children())[:21])
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.layers(input)
        target_features = self.layers(target)
        loss = nn.L1Loss()(input_features, target_features)
        return loss


def train_one_epoch(
    generator, discriminator,
    optimizer_G, optimizer_D,
    criterion_GAN, criterion_reconstruction, perceptual_loss,
    dataloader, dataloader_val, epoch, device, writer,
    early_stopping, ssim_metric, fid_metric,  # Added fid_metric
    lambda_perceptual=10.0, lambda_reconstruction=10.0
):
    generator.train()
    discriminator.train()

    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        input_image = batch['image'].to(device)         # (B, 3, H, W)
        wall_mask = batch['mask'].to(device)            # (B, 1, H, W)
        target_color = batch['target_color'].to(device) # (B, 3)

        batch_size, _, height, width = input_image.size()

        # Generator forward pass
        fake_image = generator(input_image, wall_mask, target_color)

        # Prepare target color map for loss calculation
        target_color_map = target_color.view(batch_size, 3, 1, 1).expand(batch_size, 3, height, width)

        ## Train Discriminator ##
        optimizer_D.zero_grad()

        # Real images
        pred_real = discriminator(input_image, input_image, wall_mask)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))

        # Fake images
        pred_fake = discriminator(input_image, fake_image.detach(), wall_mask)
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))

        # Total discriminator loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        ## Train Generator ##
        optimizer_G.zero_grad()

        # Adversarial loss
        pred_fake = discriminator(input_image, fake_image, wall_mask)
        loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))

        # Perceptual loss (applied only on wall regions)
        loss_perceptual = perceptual_loss(fake_image * wall_mask, input_image * wall_mask)

        # Reconstruction loss (compare wall regions to target color)
        loss_reconstruction = criterion_reconstruction(
            fake_image * wall_mask, target_color_map * wall_mask)

        # Total generator loss
        loss_G = loss_G_GAN + lambda_perceptual * loss_perceptual + lambda_reconstruction * loss_reconstruction
        loss_G.backward()
        optimizer_G.step()

        # Logging
        global_step = epoch * len(dataloader) + i
        writer.add_scalar('Loss/G', loss_G.item(), global_step)
        writer.add_scalar('Loss/D', loss_D.item(), global_step)
        writer.add_scalar('Loss/G_GAN', loss_G_GAN.item(), global_step)
        writer.add_scalar('Loss/Perceptual', loss_perceptual.item(), global_step)
        writer.add_scalar('Loss/Reconstruction', loss_reconstruction.item(), global_step)

        # Save sample images every N iterations
        if i % 300 == 0:
            save_sample_images(input_image, fake_image, wall_mask, target_color, epoch, i, writer, global_step)
    
    # Validation after each epoch
    val_metrics = validate(generator, discriminator, criterion_GAN, criterion_reconstruction, perceptual_loss, dataloader_val, device, ssim_metric, fid_metric, lambda_reconstruction, lambda_perceptual)
    
    # Log validation metrics
    writer.add_scalar('Val/Loss/G', val_metrics['avg_loss_G'], epoch)
    writer.add_scalar('Val/Loss/D', val_metrics['avg_loss_D'], epoch)
    writer.add_scalar('Val/Loss/G_GAN', val_metrics['avg_loss_G_GAN'], epoch)
    writer.add_scalar('Val/Loss/Reconstruction', val_metrics['avg_loss_reconstruction'], epoch)
    writer.add_scalar('Val/Loss/Perceptual', val_metrics['avg_loss_perceptual'], epoch)
    writer.add_scalar('Val/SSIM', val_metrics['avg_ssim'], epoch)
    writer.add_scalar('Val/FID', val_metrics['avg_fid'], epoch)  # Uncomment if FID is computed

    print(f"Epoch {epoch} Validation Metrics: {val_metrics}")

    # Implement Early Stopping based on avg_ssim
    early_stopping(val_metrics['avg_ssim'], generator)
    
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        return True  # Indicate to stop training

    return False  # Continue training


def save_sample_images(input_image, fake_image, wall_mask, target_color, epoch, batch_idx, writer, global_step):
    def denormalize(tensor):
        return (tensor + 1) / 2

    input_image = denormalize(input_image.cpu())
    fake_image = denormalize(fake_image.cpu())
    wall_mask = wall_mask.cpu()
    target_color = target_color.cpu()

    # Create target color images
    batch_size = target_color.size(0)
    height, width = input_image.size(2), input_image.size(3)
    target_color_images = target_color.view(batch_size, 3, 1, 1).expand(batch_size, 3, height, width)
    target_color_images = denormalize(target_color_images)

    # Convert mask to 3 channels for visualization
    wall_mask_vis = wall_mask.repeat(1, 3, 1, 1)

    # Log images to TensorBoard
    writer.add_images(f'Epoch_{epoch}/Input', input_image, global_step)
    writer.add_images(f'Epoch_{epoch}/Generated', fake_image, global_step)
    writer.add_images(f'Epoch_{epoch}/Wall_Mask', wall_mask_vis, global_step)
    writer.add_images(f'Epoch_{epoch}/Target_Color', target_color_images, global_step)

    # Save images to disk (optional)
    save_images_folder = os.path.join('saved_images', f'epoch_{epoch}')
    os.makedirs(save_images_folder, exist_ok=True)

    save_image(input_image, os.path.join(save_images_folder, f'input_epoch_{epoch}_batch_{batch_idx}.png'))
    save_image(fake_image, os.path.join(save_images_folder, f'generated_epoch_{epoch}_batch_{batch_idx}.png'))
    save_image(wall_mask_vis, os.path.join(save_images_folder, f'mask_epoch_{epoch}_batch_{batch_idx}.png'))
    save_image(target_color_images, os.path.join(save_images_folder, f'target_color_epoch_{epoch}_batch_{batch_idx}.png'))


def validate(generator, discriminator, criterion_GAN, criterion_reconstruction, perceptual_loss, dataloader_val, device, ssim_metric, fid_metric, lambda_reconstruction=10.0, lambda_perceptual=10.0):
    """
    Runs validation on the validation dataset and computes average metrics.
    """
    generator.eval()
    discriminator.eval()
    
    total_loss_G = 0.0
    total_loss_D = 0.0
    total_loss_G_GAN = 0.0
    total_loss_reconstruction = 0.0
    total_loss_perceptual = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader_val, desc='Validation'):
            input_image = batch['image'].to(device)         # (B, 3, H, W)
            wall_mask = batch['mask'].to(device)            # (B, 1, H, W)
            target_color = batch['target_color'].to(device) # (B, 3)

            batch_size, _, height, width = input_image.size()

            # Generator forward pass
            fake_image = generator(input_image, wall_mask, target_color)

            # Prepare target color map for loss calculation
            target_color_map = target_color.view(batch_size, 3, 1, 1).expand(batch_size, 3, height, width)

            ## Discriminator Forward Pass ##
            # Real images
            pred_real = discriminator(input_image, input_image, wall_mask)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))

            # Fake images
            pred_fake = discriminator(input_image, fake_image, wall_mask)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))

            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            total_loss_D += loss_D.item()

            ## Generator Forward Pass ##
            # Adversarial loss
            pred_fake_for_G = discriminator(input_image, fake_image, wall_mask)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G).to(device))

            # Perceptual loss (compare generated wall regions with original wall regions)
            loss_perceptual = perceptual_loss(fake_image * wall_mask, input_image * wall_mask)

            # Reconstruction loss (compare generated wall regions with target color map)
            loss_reconstruction = criterion_reconstruction(
                fake_image * wall_mask, target_color_map * wall_mask)

            # Total generator loss
            loss_G = loss_G_GAN + lambda_perceptual * loss_perceptual + lambda_reconstruction * loss_reconstruction
            total_loss_G += loss_G.item()
            total_loss_G_GAN += loss_G_GAN.item()
            total_loss_reconstruction += loss_reconstruction.item()
            total_loss_perceptual += loss_perceptual.item()

            # Compute SSIM using torchmetrics
            ssim_val = compute_ssim(fake_image, input_image, wall_mask, ssim_metric)
            total_ssim += ssim_val

            # Update FID metric
            # Rescale images from [-1, 1] to [0, 1]
            fake_image_rescaled = (fake_image + 1) / 2
            real_image_rescaled = (input_image + 1) / 2

            # Ensure the images are of type float32
            fake_image_rescaled = fake_image_rescaled.float()
            real_image_rescaled = real_image_rescaled.float()

            # Update FID metric
            fid_metric.update(fake_image_rescaled, real=False)  # fake images
            fid_metric.update(real_image_rescaled, real=True)   # real images

    # Compute FID score
    fid_score = fid_metric.compute().item()

    # Reset FID metric for next epoch
    fid_metric.reset()

    # Compute average metrics
    avg_loss_G = total_loss_G / len(dataloader_val)
    avg_loss_D = total_loss_D / len(dataloader_val)
    avg_loss_G_GAN = total_loss_G_GAN / len(dataloader_val)
    avg_loss_reconstruction = total_loss_reconstruction / len(dataloader_val)
    avg_loss_perceptual = total_loss_perceptual / len(dataloader_val)
    avg_ssim = total_ssim / len(dataloader_val)

    return {
        'avg_loss_G': avg_loss_G,
        'avg_loss_D': avg_loss_D,
        'avg_loss_G_GAN': avg_loss_G_GAN,
        'avg_loss_reconstruction': avg_loss_reconstruction,
        'avg_loss_perceptual': avg_loss_perceptual,
        'avg_ssim': avg_ssim,
        'avg_fid': fid_score
    }


def compute_ssim(fake_image, real_image, wall_mask, ssim_metric):
    # Apply wall mask
    fake_wall = fake_image * wall_mask
    real_wall = real_image * wall_mask
    
    # Compute SSIM
    ssim_val = ssim_metric(fake_wall, real_wall)
    return ssim_val.item()


