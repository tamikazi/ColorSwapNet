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

        self.transform = transform if transform else transforms.ToTensor()
        self.target_colors = target_colors

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)

        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0).float()  # Convert to binary mask

        # Get target color
        if self.target_colors:
            target_color = self.target_colors[idx % len(self.target_colors)]
        else:
            target_color = torch.rand(3)

        sample = {
            'image': image,
            'mask': mask,
            'target_color': target_color,
            'image_path': self.image_paths[idx]
        }

        return sample
