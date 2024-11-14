# models/models_gan/generator.py

import torch
import torch.nn as nn

class UNetDown(nn.Module):
    """Downsampling layer in U-Net architecture"""
    def __init__(self, in_size, out_size, normalize=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """Upsampling layer in U-Net architecture"""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # Concatenate with skip connection
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    def __init__(self, input_channels=7, output_channels=3):
        super(Generator, self).__init__()
        # Input channels: 3 (image) + 1 (mask) + 3 (target_color_map) = 7

        # Downsampling layers (Encoder)
        self.down1 = UNetDown(input_channels, 64, normalize=False)  # [B, 64, 128, 128]
        self.down2 = UNetDown(64, 128)                              # [B, 128, 64, 64]
        self.down3 = UNetDown(128, 256)                             # [B, 256, 32, 32]
        self.down4 = UNetDown(256, 512)                             # [B, 512, 16, 16]
        self.down5 = UNetDown(512, 512)                             # [B, 512, 8, 8]
        self.down6 = UNetDown(512, 512)                             # [B, 512, 4, 4]
        self.down7 = UNetDown(512, 512)                             # [B, 512, 2, 2]
        self.down8 = UNetDown(512, 512, normalize=False)            # [B, 512, 1, 1]

        # Upsampling layers (Decoder)
        self.up1 = UNetUp(512, 512, dropout=0.5)                    # [B, 1024, 2, 2]
        self.up2 = UNetUp(1024, 512, dropout=0.5)                   # [B, 1024, 4, 4]
        self.up3 = UNetUp(1024, 512, dropout=0.5)                   # [B, 1024, 8, 8]
        self.up4 = UNetUp(1024, 512, dropout=0.0)                   # [B, 1024, 16, 16]
        self.up5 = UNetUp(1024, 256, dropout=0.0)                   # [B, 512, 32, 32]
        self.up6 = UNetUp(512, 128, dropout=0.0)                    # [B, 256, 64, 64]
        self.up7 = UNetUp(256, 64, dropout=0.0)                     # [B, 128, 128, 128]

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, image, mask, target_color):
        # Expand target color to match spatial dimensions
        B, _, H, W = image.size()
        target_color_map = target_color.view(B, 3, 1, 1).expand(B, 3, H, W)
        # Prepare input: concatenate image, mask, and target color map
        gen_input = torch.cat([image, mask, target_color_map], dim=1)  # Shape: [B, 7, H, W]

        # Encoder path
        d1 = self.down1(gen_input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder path with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        # Final output layer
        output = self.final(u7)  # Output shape: [B, 3, H, W]

        # Apply mask to output
        output = image * (1 - mask) + output * mask

        return output
