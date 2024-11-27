# models/models_cyclegan/generator.py

import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False, use_norm=True):
        super(UNetBlock, self).__init__()
        self.down = down
        if down:
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if use_norm:
                layers.append(nn.InstanceNorm2d(out_channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if use_norm:
                layers.append(nn.InstanceNorm2d(out_channels, affine=True))
            layers.append(nn.ReLU(inplace=True))
            if use_dropout:
                layers.append(nn.Dropout(0.5))
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        return self.conv(x)

class WallColorGenerator(nn.Module):
    def __init__(self, input_nc=7, output_nc=3, ngf=64):
        super(WallColorGenerator, self).__init__()
        # Encoder
        self.down1 = UNetBlock(input_nc, ngf, down=True, use_norm=True)
        self.down2 = UNetBlock(ngf, ngf * 2, down=True, use_norm=True)
        self.down3 = UNetBlock(ngf * 2, ngf * 4, down=True, use_norm=True)
        self.down4 = UNetBlock(ngf * 4, ngf * 8, down=True, use_norm=True)
        self.down5 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=True)
        self.down6 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=True)
        self.down7 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=True)
        self.down8 = UNetBlock(ngf * 8, ngf * 8, down=True, use_norm=False)  # Disable norm here

        # Decoder
        self.up1 = UNetBlock(ngf * 8, ngf * 8, down=False, use_dropout=True, use_norm=False)  # Disable norm here
        self.up2 = UNetBlock(ngf * 8 * 2, ngf * 8, down=False, use_dropout=True, use_norm=True)
        self.up3 = UNetBlock(ngf * 8 * 2, ngf * 8, down=False, use_dropout=True, use_norm=True)
        self.up4 = UNetBlock(ngf * 8 * 2, ngf * 8, down=False, use_dropout=False, use_norm=True)
        self.up5 = UNetBlock(ngf * 8 * 2, ngf * 4, down=False, use_dropout=False, use_norm=True)
        self.up6 = UNetBlock(ngf * 4 * 2, ngf * 2, down=False, use_dropout=False, use_norm=True)
        self.up7 = UNetBlock(ngf * 2 * 2, ngf, down=False, use_dropout=False, use_norm=True)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_image, wall_mask, target_color):
        # Prepare target color map
        batch_size, _, height, width = input_image.size()
        target_color_map = target_color.view(batch_size, 3, 1, 1).expand(batch_size, 3, height, width)
        # Concatenate inputs
        generator_input = torch.cat([input_image, wall_mask, target_color_map], dim=1)
        # Encoder
        d1 = self.down1(generator_input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)
        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)
        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)
        output = self.up8(u7)
        # Apply the wall mask to ensure only wall regions are modified
        final_output = input_image * (1 - wall_mask) + output * wall_mask
        return final_output
