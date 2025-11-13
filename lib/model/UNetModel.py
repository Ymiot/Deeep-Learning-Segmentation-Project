import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64, depth=4):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for d in range(depth):
            outc = base * (2**d)
            self.downs.append(double_conv(prev, outc))
            self.pools.append(nn.MaxPool2d(2))
            prev = outc
        self.bottleneck = double_conv(prev, prev*2)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for d in reversed(range(depth)):
            inc = prev*2
            outc = base * (2**d)
            self.ups.append(nn.ConvTranspose2d(inc, outc, 2, stride=2))
            self.up_convs.append(double_conv(inc, outc))
            prev = outc
        self.final = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for i, (up, up_conv) in enumerate(zip(self.ups, self.up_convs)):
            x = up(x)
            skip = skips[-(i+1)]
            if x.shape[-2:] != skip.shape[-2:]:
                # safe center crop
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                skip = skip[:, :, diffY//2:skip.size(2)-diffY//2, diffX//2:skip.size(3)-diffX//2]
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)
        return self.final(x)
