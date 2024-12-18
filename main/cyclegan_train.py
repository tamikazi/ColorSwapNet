# main/cyclegan_train.py

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress INFO messages
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
# Import SSIM from torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure
from datasets.dataset_cyclegan import CycleGANDataset
from models.models_cyclegan.generator import WallColorGenerator
from models.models_cyclegan.discriminator import WallColorDiscriminator
from src.src_cyclegan.train import train_one_epoch, PerceptualLoss
from src.src_cyclegan.train import save_sample_images  # Make sure to import if in separate file

from src.src_cyclegan.utils import EarlyStopping  # Import the EarlyStopping class

from torchmetrics.image.fid import FrechetInceptionDistance

from utils.constants import ROOT_DATASET, DEVICE, NUM_WORKERS

import itertools
from torchvision import models
import torch.nn as nn


def main():
    # Directories
    TRAIN_IMAGE_ROOT = os.path.join(ROOT_DATASET, "images/training")
    TRAIN_MASK_ROOT = os.path.join(ROOT_DATASET, "annotations/training")
    VAL_IMAGE_ROOT = os.path.join(ROOT_DATASET, "images/validation")
    VAL_MASK_ROOT = os.path.join(ROOT_DATASET, "annotations/validation")

    # Create output directories if they don't exist
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('saved_images', exist_ok=True)

    # Hyperparameters
    batch_size = 8
    num_epochs = 100
    learning_rate = 0.00001
    device = DEVICE

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
    ])

    # Create dataset
    dataset = CycleGANDataset(TRAIN_IMAGE_ROOT, TRAIN_MASK_ROOT, transform=transform)
    dataset_val = CycleGANDataset(VAL_IMAGE_ROOT, VAL_MASK_ROOT, transform=transform)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize models
    generator = WallColorGenerator().to(device)
    discriminator = WallColorDiscriminator().to(device)

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_reconstruction = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate * 2, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate * 0.01, betas=(0.5, 0.999))

    # Learning rate schedulers (optional)
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)

    # Initialize or load checkpoint
    start_epoch = 1  # Default start epoch

    # Check if a checkpoint exists
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=15, verbose=True, delta=0.001, path='best_generator.pt', mode='max')

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/cyclegan_training')

    # Initialize SSIM metric from torchmetrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)  # Since images are normalized to [-1, 1]

    # Initialize FID metric from torchmetrics
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)  # Default feature layer

    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        stop = train_one_epoch(
            generator, discriminator,
            optimizer_G, optimizer_D,
            criterion_GAN, criterion_reconstruction, perceptual_loss,
            dataloader, dataloader_val, epoch, device, writer,
            early_stopping, ssim_metric, fid_metric,  # Pass fid_metric
            lambda_perceptual=10.0, lambda_reconstruction=10.0
        )
        
        if stop:
            print("Early stopping triggered.")
            break

        # Load the best generator saved by EarlyStopping
        generator.load_state_dict(torch.load('best_generator.pt'))
        generator.to(device)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        # Save models and optimizer states
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))

        # Optionally, save a checkpoint with the epoch number
        # torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

    # Close the writer
    writer.close()


if __name__ == '__main__':
    main()
