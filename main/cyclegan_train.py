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
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('output_images', exist_ok=True)

    # Hyperparameters
    batch_size = 1
    num_epochs = 100
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize models
    G_AB = ResnetGenerator().to(device)
    G_BA = ResnetGenerator().to(device)
    D_A = NLayerDiscriminator().to(device)
    D_B = NLayerDiscriminator().to(device)

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/cyclegan_training')

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        loss_G, loss_D_A, loss_D_B = train_one_epoch(
            G_AB, G_BA, D_A, D_B,
            optimizer_G, optimizer_D_A, optimizer_D_B,
            criterion_GAN, criterion_cycle, criterion_identity,
            dataloader, epoch, device, writer
        )

        # Save models
        torch.save(G_AB.state_dict(), f'checkpoints/G_AB_epoch_{epoch}.pth')
        torch.save(G_BA.state_dict(), f'checkpoints/G_BA_epoch_{epoch}.pth')
        torch.save(D_A.state_dict(), f'checkpoints/D_A_epoch_{epoch}.pth')
        torch.save(D_B.state_dict(), f'checkpoints/D_B_epoch_{epoch}.pth')

    # Close the writer
    writer.close()

if __name__ == '__main__':
    main()
