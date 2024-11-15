# models/models_gan/discriminator.py

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=6):
        super(Discriminator, self).__init__()
        # Input channels: 3 (image) + 3 (target color map) = 6

        self.model = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 128, 128]
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv Layer 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Layer
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # [B, 1, 13, 13]
            # No activation function here
        )

    def forward(self, image, target_color):
        B, _, H, W = image.size()
        # Expand target color to match spatial dimensions
        target_color_map = target_color.view(B, 3, 1, 1).expand(B, 3, H, W)
        # Concatenate image and target color map
        x = torch.cat([image, target_color_map], dim=1)  # Shape: [B, 6, H, W]
        # Pass through the network
        x = self.model(x)
        return x  # Output logits
