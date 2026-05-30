import torch
import torch.nn as nn
import torch.nn.functional as F

class Convo(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # Skip connection logic
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

        groups_in = min(32, in_channels)
        groups_out = min(32, out_channels)

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups_in, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups_out, num_channels=out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=out_channels,
            ),
        )

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()
        # Encoder (Downsampling Path)
        self.down1 = Convo(in_channels, features)
        self.pool1 = nn.Conv3d(features, features, kernel_size=3, stride=2, padding=1)

        self.down2 = Convo(features, features * 2)
        self.pool2 = nn.Conv3d(features * 2, features * 2, kernel_size=3, stride=2, padding=1)