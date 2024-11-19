# main/cyclegan_train.py

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress INFO messages
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.dataset_cyclegan import CycleGANDataset
from models.models_cyclegan.generator import ResnetGenerator
from models.models_cyclegan.discriminator import NLayerDiscriminator
from src.src_cyclegan.train import train_one_epoch

from utils.constants import ROOT_DATASET, DEVICE, NUM_WORKERS

import itertools


def main():
    # Directories
    IMAGE_ROOT = ROOT_DATASET + "/images/training"
    MASK_ROOT = ROOT_DATASET + "/annotations/training"

    # Create output directories if they don't exist
    checkpoint_dir = 'checkpoints'
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('output_images', exist_ok=True)

    # Hyperparameters
    batch_size = 4
    num_epochs = 200
    learning_rate = 0.0002
    device = DEVICE

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
    ])

    # Create dataset
    dataset = CycleGANDataset(IMAGE_ROOT, MASK_ROOT, transform=transform)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

    # Initialize models
    G_AB = ResnetGenerator().to(device)
    G_BA = ResnetGenerator().to(device)
    D_A = NLayerDiscriminator().to(device)
    D_B = NLayerDiscriminator().to(device)

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Initialize or load checkpoint
    start_epoch = 1  # Default start epoch

    # Check if a checkpoint exists
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
        G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
        D_A.load_state_dict(checkpoint['D_A_state_dict'])
        D_B.load_state_dict(checkpoint['D_B_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/cyclegan_training')

    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        loss_G, loss_D_A, loss_D_B = train_one_epoch(
            G_AB, G_BA, D_A, D_B,
            optimizer_G, optimizer_D_A, optimizer_D_B,
            criterion_GAN, dataloader, epoch, device, writer
        )

        # Save models and optimizer states
        checkpoint = {
            'epoch': epoch,
            'G_AB_state_dict': G_AB.state_dict(),
            'G_BA_state_dict': G_BA.state_dict(),
            'D_A_state_dict': D_A.state_dict(),
            'D_B_state_dict': D_B.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))

        # Optionally, save a checkpoint with the epoch number
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

    # Close the writer
    writer.close()

if __name__ == '__main__':
    main()
