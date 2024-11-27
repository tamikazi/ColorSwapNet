# models/models_cyclegan/discriminator.py

import torch
import torch.nn as nn

class WallColorDiscriminator(nn.Module):
    def __init__(self, input_nc=7, ndf=64):
        super(WallColorDiscriminator, self).__init__()
        # Input channels: 3 (input image) + 3 (generated/real image) + 1 (wall mask) = 7
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            # Output: (bs, 1, H/8, W/8)
        )

    def forward(self, input_image, target_image, wall_mask):
        # Ensure wall_mask has the same spatial dimensions as input_image
        if wall_mask.size(1) == 1 and input_image.size(1) == 3:
            wall_mask = wall_mask.expand(-1, 1, input_image.size(2), input_image.size(3))
        discriminator_input = torch.cat([input_image, target_image, wall_mask], dim=1)  # (bs, 7, H, W)
        output = self.model(discriminator_input)
        return output
