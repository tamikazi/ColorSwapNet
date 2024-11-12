# datasets/dataset_gan.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class GANDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None, target_colors=None):
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
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.target_colors = target_colors  # List of target colors (list of RGB tuples in range [0,1])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Apply transformations
        image = self.transform(image)
        mask = self.mask_transform(mask)

        # Invert mask: walls (value 0) become 1, others become 0
        mask = (mask == 0).float()

        # Get target color
        if self.target_colors:
            target_color = self.target_colors[idx % len(self.target_colors)]
            target_color = torch.tensor(target_color).float()  # Ensure it's a tensor
        else:
            # Generate a random target color (RGB values between 0 and 1)
            target_color = torch.rand(3)

        sample = {
            'image': image,
            'mask': mask,
            'target_color': target_color,
            'image_path': self.image_paths[idx]
        }

        return sample
