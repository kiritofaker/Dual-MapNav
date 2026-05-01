"""BEV (Bird's-Eye View) Encoder for exploration nodes."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVEncoder(nn.Module):
    """
    Encodes bird's-eye view maps representing traversable areas.

    The BEV representation captures:
    - Occupancy grid (traversable vs blocked)
    - Distance transform to obstacles
    - Gradient/orientation information
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        out_channels: int = 512,
        img_size: int = 64
    ):
        super().__init__()

        self.img_size = img_size

        # Convolutional encoder for BEV maps
        self.encoder = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),

            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),

            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling to handle variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Output projection
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 4 * 16, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_map: (B, C, H, W) bird's-eye view map

        Returns:
            bev_features: (B, D) encoded BEV features
        """
        x = self.encoder(bev_map)
        x = self.pool(x)
        x = self.projection(x)
        return x


class BEVFeaturePyramid(nn.Module):
    """
    Multi-scale BEV feature extraction for hierarchical map representation.
    Used for capturing both local and global context.
    """

    def __init__(self, hidden_channels: int = 128, out_channels: int = 512):
        super().__init__()

        # Backbone at different scales
        self.backbone = nn.ModuleDict({
            '32': nn.Sequential(
                nn.Conv2d(3, hidden_channels, 3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ),
            '16': nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
                nn.BatchNorm2d(hidden_channels * 2),
                nn.ReLU(inplace=True),
            ),
            '8': nn.Sequential(
                nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, padding=1),
                nn.BatchNorm2d(hidden_channels * 4),
                nn.ReLU(inplace=True),
            ),
        })

        # FPN-style fusion
        self.lateral = nn.ModuleDict({
            '16': nn.Conv2d(hidden_channels * 2, out_channels, 1),
            '8': nn.Conv2d(hidden_channels * 4, out_channels, 1),
        })

        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, bev_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_map: (B, 3, H, W) BEV map

        Returns:
            pyramid_features: (B, out_channels, H/8, W/8) multi-scale features
        """
        features = {}

        # Build pyramid
        x = bev_map
        for scale, layer in self.backbone.items():
            x = layer(x)
            features[scale] = x
            if scale != '8':
                x = F.max_pool2d(x, 2)

        # Top-down pathway with lateral connections
        c4 = self.lateral['8'](features['8'])

        c3 = F.interpolate(c4, size=features['8'].shape[2:], mode='bilinear', align_corners=False)
        c3 = c3 + self.lateral['8'](features['8'])

        output = self.output_conv(c3)
        return output
