"""Main Observation Encoder combining RGB, Depth, and BEV."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .rgb_encoder import RGBEncoder, CLIPVisionEncoder
from .depth_encoder import DepthEncoder, DepthHistogramEncoder
from .cross_modal_fusion import CrossModalFusion, AttentionFusion


class ObservationEncoder(nn.Module):
    """
    Full observation encoder for VLN.

    Processes:
    - RGB images
    - Depth images
    - Bird's-eye view maps

    With cross-modal fusion for unified representation.
    """

    def __init__(
        self,
        rgb_backbone: str = "resnet50",
        use_clip: bool = False,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_fusion_layers: int = 4,
        dropout: float = 0.1,
        bev_channels: int = 3,
        bev_size: int = 64
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # RGB encoder
        if use_clip:
            self.rgb_encoder = CLIPVisionEncoder(
                model_name="ViT-L/14",
                output_dim=embed_dim
            )
        else:
            self.rgb_encoder = RGBEncoder(
                backbone=rgb_backbone,
                output_dim=embed_dim
            )

        # Depth encoder
        self.depth_encoder = DepthEncoder(
            input_channels=1,
            output_dim=embed_dim
        )

        # BEV encoder (reuses map representation encoder)
        from models.map_representation import BEVEncoder
        self.bev_encoder = BEVEncoder(
            in_channels=bev_channels,
            out_channels=embed_dim,
            img_size=bev_size
        )

        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_fusion_layers
        )

        # Temporal encoding for sequence of observations
        self.temporal_encoder = TemporalEncoder(embed_dim)

    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        bev: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb: (B, 3, H, W) RGB image
            depth: (B, 1, H, W) or (B, H, W) depth image
            bev: (B, C, H, W) bird's-eye view map
            return_dict: Whether to return dict or tuple

        Returns:
            Dictionary or tuple with encoded features
        """
        outputs = {}

        # Encode each modality
        if rgb is not None:
            rgb_features = self.rgb_encoder(rgb)
            outputs['rgb'] = rgb_features

        if depth is not None:
            depth_features = self.depth_encoder(depth)
            outputs['depth'] = depth_features

        if bev is not None:
            bev_features = self.bev_encoder(bev)
            outputs['bev'] = bev_features

        # Cross-modal fusion if multiple modalities present
        if len(outputs) >= 2:
            rgb_feat = outputs.get('rgb', torch.zeros(1, self.embed_dim, device=rgb.device if rgb is not None else 'cpu'))
            depth_feat = outputs.get('depth', torch.zeros(1, self.embed_dim, device=rgb.device if rgb is not None else 'cpu'))
            bev_feat = outputs.get('bev', torch.zeros(1, self.embed_dim, device=rgb.device if rgb is not None else 'cpu'))

            _, _, _, fused = self.fusion(rgb_feat, depth_feat, bev_feat)
            outputs['fused'] = fused

        if return_dict:
            return outputs
        else:
            # Return fused or first available
            return outputs.get('fused', outputs.get('rgb', outputs.get('depth', None)))


class TemporalEncoder(nn.Module):
    """
    Encodes temporal sequences of observations.
    Useful for processing a sequence of past observations.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Positional encoding for time
        self.time_pos = PositionalEncoding1D(embed_dim)

        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (B, T, D) sequence of observation features

        Returns:
            encoded: (B, D) temporally encoded observation
        """
        # Add temporal position
        x = self.time_pos(observations)

        # Transformer encoding
        x = self.transformer(x)

        # Pool over time
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pool(x).squeeze(-1)  # (B, D)

        return x


class PositionalEncoding1D(nn.Module):
    """1D positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) sequence

        Returns:
            x with positional encoding
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class Ego3DObservationEncoder(nn.Module):
    """
    Ego3D-style observation encoder.
    Projects 3D ego-centric observations into a common feature space.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_slices: int = 8
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_slices = num_slices

        # RGB slice encoder (processes horizontal slices)
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(3 * num_slices, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embed_dim)
        )

        # Depth slice encoder
        self.depth_slice_encoder = nn.Sequential(
            nn.Conv2d(num_slices, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embed_dim)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) RGB image
            depth: (B, 1, H, W) depth image

        Returns:
            ego_features: (B, D) ego-centric 3D features
        """
        B, C, H, W = rgb.shape

        # Create horizontal slices
        slice_size = H // self.num_slices
        rgb_slices = []
        depth_slices = []

        for i in range(self.num_slices):
            start = i * slice_size
            end = (i + 1) * slice_size
            rgb_slice = rgb[:, :, start:end, :]  # (B, 3, H/num_slices, W)
            depth_slice = depth[:, :, start:end, :]  # (B, 1, H/num_slices, W)

            # Expand depth to 3 channels
            depth_slice_expanded = depth_slice.expand(-1, 3, -1, -1)

            # Concatenate along slice dimension
            rgb_slices.append(rgb_slice)
            depth_slices.append(depth_slice_expanded)

        # Stack slices
        rgb_stacked = torch.cat(rgb_slices, dim=2)  # (B, 3*num_slices, H/num_slices, W)
        depth_stacked = torch.cat(depth_slices, dim=2)  # (B, 3*num_slices, H/num_slices, W)

        # Encode
        rgb_features = self.slice_encoder(rgb_stacked)
        depth_features = self.depth_slice_encoder(depth_stacked[:, :self.num_slices, :, :])

        # Fuse
        fused = self.fusion(torch.cat([rgb_features, depth_features], dim=-1))

        return fused
