import torch
from torch import nn
import torch.nn.functional as F


class LaplacianPyramid(nn.Module):
    def __init__(self, channels=3, scales=(1.0, 0.5, 0.25, 0.125)):
        super().__init__()
        self.channels = channels
        self.scales = scales
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    @property
    def out_channels(self):
        return self.channels * len(self.scales)

    def forward(self, x):
        _, _, h, w = x.shape
        laps = []
        for scale in self.scales:
            if scale != 1.0:
                scaled = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
            else:
                scaled = x
            lap = F.conv2d(scaled, self.kernel, padding=1, groups=self.channels)
            if scale != 1.0:
                lap = F.interpolate(lap, size=(h, w), mode="bilinear", align_corners=False)
            laps.append(lap)
        return torch.cat(laps, dim=1)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + res)


class RDNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, base_channels=32, num_blocks=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        blocks = [ResBlock(base_channels) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, 3, padding=1),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return torch.sigmoid(x)
