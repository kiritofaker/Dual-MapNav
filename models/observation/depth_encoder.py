"""Depth Encoder for observation processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthEncoder(nn.Module):
    """
    Encodes depth images from robot observations.
    Processes depth as a single-channel image with spatial encoding.
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 256,
        output_dim: int = 512
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            # Layer 1: downsample spatial dimension
            nn.Conv2d(input_channels, hidden_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Global pooling and projection
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, H, W) or (B, H, W) depth image

        Returns:
            features: (B, D) encoded depth features
        """
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        x = self.encoder(depth)
        x = self.projection(x)
        return x


class DepthHistogramEncoder(nn.Module):
    """
    Encodes depth using histogram binning for better spatial awareness.
    Creates a multi-channel representation of depth values.
    """

    def __init__(
        self,
        num_bins: int = 32,
        hidden_channels: int = 128,
        output_dim: int = 512
    ):
        super().__init__()

        self.num_bins = num_bins

        # Bin edges (can be learned)
        self.bin_edges = nn.Parameter(
            torch.linspace(0, 10, num_bins + 1),
            requires_grad=False
        )

        # Encoder for histogram-like depth
        self.encoder = nn.Sequential(
            nn.Conv2d(num_bins, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, output_dim),
            nn.LayerNorm(output_dim),
        )

    def depth_to_histogram(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Convert depth to histogram representation.

        Args:
            depth: (B, 1, H, W) depth values

        Returns:
            histogram: (B, num_bins, H, W) soft histogram
        """
        B, _, H, W = depth.shape
        depth = depth.expand(-1, self.num_bins, -1, -1)

        # Create bin centers
        bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
        bin_centers = bin_centers.view(1, -1, 1, 1).to(depth.device)

        # Compute soft assignment to bins
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        diff = torch.abs(depth - bin_centers)
        histogram = torch.exp(-diff / (bin_width / 2))

        # Normalize
        histogram = histogram / histogram.sum(dim=1, keepdim=True)

        return histogram

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, H, W) depth image

        Returns:
            features: (B, D) encoded depth features
        """
        hist = self.depth_to_histogram(depth)
        features = self.encoder(hist)
        return features


class MiDaSDepthEncoder(nn.Module):
    """
    Wrapper for MiDaS depth estimation model.
    Provides metric depth from RGB images.
    """

    def __init__(
        self,
        model_name: str = "MiDaS_v3_1",
        output_dim: int = 512,
        freeze: bool = True
    ):
        super().__init__()

        self.output_dim = output_dim

        try:
            from torchvision.models import MidasNet, MidasNet_large

            if "large" in model_name.lower():
                self.midas = MidasNet_large(pretrained=True if freeze else False)
            else:
                self.midas = MidasNet(pretrained=True if freeze else False)

            if freeze:
                for param in self.midas.parameters():
                    param.requires_grad = False

            self.feature_dim = 256 if "large" not in model_name.lower() else 256

        except Exception as e:
            print(f"Warning: Could not load MiDaS: {e}")
            self.midas = None
            self.feature_dim = 256

        # Projection
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) RGB image for depth estimation

        Returns:
            features: (B, D) depth features
        """
        if self.midas is not None:
            with torch.no_grad():
                depth = self.midas(rgb)
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=rgb.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            depth = torch.randn(
                rgb.shape[0], 1, rgb.shape[2], rgb.shape[3],
                device=rgb.device
            )

        # Encode depth
        encoder = DepthEncoder(output_dim=self.output_dim)
        features = encoder(depth)
        return features
