# datasets/dataset_cyclegan.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CycleGANDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None):
        self.image_root = image_root
        self.mask_root = mask_root

        self.image_paths = sorted([os.path.join(self.image_root, f) for f in os.listdir(self.image_root)
                                   if f.endswith('.jpg') or f.endswith('.png')])
        self.mask_paths = sorted([os.path.join(self.mask_root, f) for f in os.listdir(self.mask_root)
                                  if f.endswith('.png')])

        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks should be the same."

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # Converts to [0,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask paths
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  # Shape: [3, H, W]

        # Load mask
        mask = Image.open(mask_path)
        mask = self.mask_transform(mask)  # Shape: [1, H, W]
        mask = (mask < 0.005).float()  # Adjust threshold as needed

        # Generate a random target color in [-1, 1]
        target_color = torch.rand(3) * 2 - 1  # Shape: [3]

        sample = {
            'image': image,            # Original image
            'mask': mask,              # Wall mask
            'target_color': target_color  # Random target color
        }

        return sample
