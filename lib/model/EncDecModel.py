import torch
import torch.nn as nn

class EncBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class DecBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): 
        return self.block(x)

class EncDec(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        self.enc1 = EncBlock(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = EncBlock(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = EncBlock(base*2, base*4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DecBlock(base*4, base*2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DecBlock(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))
        x = self.up2(x3)
        x = self.dec2(x)
        x = self.up1(x)
        x = self.dec1(x)
        return self.out(x)