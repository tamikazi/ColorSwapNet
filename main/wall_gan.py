# main/wall_gan.py

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress INFO messages
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.dataset_gan import GANDataset
from models.models_gan.generator import Generator
from models.models_gan.discriminator import Discriminator
from src.src_gan.train import train_one_epoch
from src.src_gan.eval import evaluate
from src.src_gan.test import test

from utils.constants import ROOT_DATASET, DEVICE, NUM_WORKERS, BATCH_PER_GPU

def main():
    # Directories
    IMAGE_TRAINING_ROOT = ROOT_DATASET + "/images/training"
    IMAGE_VALIDATION_ROOT = ROOT_DATASET + "/images/validation"
    IMAGE_TEST_ROOT = ROOT_DATASET + "/images/test"
    MASK_TRAINING_ROOT = ROOT_DATASET + "/annotations/training"
    MASK_VALIDATION_ROOT = ROOT_DATASET + "/annotations/validation"
    MASK_TEST_ROOT = ROOT_DATASET + "/annotations/test"

    # Create output directories if they don't exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('output_images', exist_ok=True)
    os.makedirs('test_output_images', exist_ok=True)

    # Hyperparameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.0002
    device = DEVICE

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
    ])

    # Create datasets
    train_dataset = GANDataset(IMAGE_TRAINING_ROOT, MASK_TRAINING_ROOT, transform=transform)
    val_dataset = GANDataset(IMAGE_VALIDATION_ROOT, MASK_VALIDATION_ROOT, transform=transform)
    test_dataset = GANDataset(IMAGE_TEST_ROOT, MASK_TEST_ROOT, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss functions
    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_L1 = torch.nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/gan_training')

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss_G, train_loss_D = train_one_epoch(
            generator, discriminator, train_loader, optimizer_G, optimizer_D,
            criterion_GAN, criterion_L1, epoch, device, writer
        )

        # Evaluate on validation set
        evaluate(generator, val_loader, epoch, device, output_dir='output_images')

        # Save models
        torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch}.pth')

    # Close the writer
    writer.close()

    # Testing after training
    print("Testing the model on the test dataset...")
    test_output_dir = 'test_output_images'
    test(generator, test_loader, device, output_dir=test_output_dir)
    print(f"Test images have been saved to {test_output_dir}")

if __name__ == '__main__':
    main()
