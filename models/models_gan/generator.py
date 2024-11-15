# models/models_gan/generator.py

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channels=7, output_channels=3):
        super(Generator, self).__init__()
        # Input channels: 3 (image) + 1 (segmented image) + 3 (target color map) = 7

        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 128, 128]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder (Upsampling)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),  # [B, 3, 256, 256]
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, image, segmented_image, target_color):
        # Expand target color to match spatial dimensions
        B, _, H, W = image.size()
        target_color_map = target_color.view(B, 3, 1, 1).expand(B, 3, H, W)
        # Prepare input: concatenate image, segmented image, and target color map
        gen_input = torch.cat([image, segmented_image, target_color_map], dim=1)  # Shape: [B, 7, H, W]

        # Encoder
        e1 = self.enc1(gen_input)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)

        # Decoder
        d1 = self.dec1(b)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        d5 = self.dec5(d4)
        output = self.final(d5)

        # Apply mask to output
        output = image * (1 - segmented_image) + output * segmented_image

        return output  # Output image in range [-1, 1]
