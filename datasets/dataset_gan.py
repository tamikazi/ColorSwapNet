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
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            # Do not convert to tensor yet
        ])

        self.target_colors = target_colors  # List of target colors (list of RGB tuples in range [0,1])

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
        # Handle different mask modes
        if mask.mode == 'RGBA':
            # If mask has an alpha channel, extract it
            alpha = mask.split()[-1]
            mask = alpha
        elif mask.mode != 'L':
            mask = mask.convert('L')

        # Apply mask transformations
        mask = self.mask_transform(mask)

        # Convert mask to numpy array
        mask_array = np.array(mask)
        # Threshold the mask to ensure binary values
        threshold = 128  # Adjust threshold if needed
        mask_array = (mask_array < threshold).astype(np.float32)  # Walls as 1, others as 0

        # Convert mask to tensor
        mask_tensor = torch.tensor(mask_array).unsqueeze(0)  # Shape: [1, H, W]

        # Ensure mask is the same size as the image
        if mask_tensor.shape[1:] != image.shape[1:]:
            mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0), size=image.shape[1:], mode='nearest').squeeze(0)

        # Get target color
        if self.target_colors:
            target_color = self.target_colors[idx % len(self.target_colors)]
            target_color = torch.tensor(target_color).float()  # Ensure it's a tensor
        else:
            # Generate a random target color (RGB values between 0 and 1)
            target_color = torch.rand(3)

        # Diagnostic: Check mask unique values
        unique_values = torch.unique(mask_tensor)
        if unique_values.numel() > 2:
            print(f"Warning: Mask at index {idx} has more than two unique values: {unique_values}")

        sample = {
            'image': image,
            'mask': mask_tensor,
            'target_color': target_color,
            'image_path': image_path
        }

        return sample
