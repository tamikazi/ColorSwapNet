# models/models_gan/generator.py

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(Generator, self).__init__()
        # Input channels: 3 (image) + 1 (mask)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Additional layers...
        )

        self.decoder = nn.Sequential(
            # Additional layers...
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output values between -1 and 1 if normalized
        )

    def forward(self, image, mask, target_color):
        # Expand target color to match spatial dimensions
        target_color_map = target_color.unsqueeze(2).unsqueeze(3).expand(-1, -1, image.size(2), image.size(3))
        # Apply mask to target color map
        target_color_applied = mask * target_color_map

        # Prepare input: concatenate image and mask
        gen_input = torch.cat([image, mask], dim=1)  # Shape: [B, 4, H, W]

        # Pass through the network
        x = self.encoder(gen_input)
        x = self.decoder(x)

        # Apply target color to the wall regions
        output = image * (1 - mask) + x * mask

        return output
