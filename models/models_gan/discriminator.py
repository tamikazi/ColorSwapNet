# models/models_gan/discriminator.py

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=6):
        super(Discriminator, self).__init__()
        # Input channels: 3 (image) + 3 (target color map)
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Additional layers...
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Additional layers...
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, image, target_color):
        # Expand target color to match spatial dimensions
        target_color_map = target_color.unsqueeze(2).unsqueeze(3).expand(-1, -1, image.size(2), image.size(3))

        # Concatenate image and target color map
        x = torch.cat([image, target_color_map], dim=1)  # Shape: [B, 6, H, W]

        # Pass through the network
        x = self.model(x)

        return x
