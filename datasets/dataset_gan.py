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
            transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor(),  # Converts to [0,1]
        ])

        self.target_colors = target_colors  # List of target colors (list of RGB tuples in range [-1,1])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask paths
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  # Shape: [3, H, W]

        # Load mask (segmented image)
        mask = Image.open(mask_path)

        # Apply mask transformations
        mask = self.mask_transform(mask)  # Shape: [1, H, W]
        # Process the mask to be binary: walls as 1, others as 0
        mask = (mask == 0).float()  # Adjust this based on your mask values

        # Get target color
        if self.target_colors:
            target_color = self.target_colors[idx % len(self.target_colors)]
            target_color = torch.tensor(target_color).float()  # Shape: [3]
        else:
            # Generate a random target color (RGB values between -1 and 1)
            target_color = torch.rand(3) * 2 - 1  # Normalize to [-1, 1]

        sample = {
            'image': image,
            'segmented_image': mask,
            'target_color': target_color,
            'image_path': image_path
        }

        return sample
