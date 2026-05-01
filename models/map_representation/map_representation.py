"""Main MapRepresentation module combining BEV, Semantic, and Text features."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict

from .exploration_nodes import BEVEncoder
from .semantic_nodes import SemanticEncoder, SemanticQueryEncoder
from .text_embeddings import TextEmbeddingLayer, CLIPTextEncoder


@dataclass
class MapRepresentationOutput:
    """Output container for map representation."""
    bev_features: torch.Tensor      # (B, D) exploration features
    semantic_features: torch.Tensor # (B, D) semantic features
    text_embeddings: torch.Tensor   # (B, N, D) or (B, D) text embeddings
    combined_features: torch.Tensor # (B, D) combined representation


class MapRepresentation(nn.Module):
    """
    Unified map representation with three node types:

    1. Exploration Nodes: Traversable area features (BEV)
    2. Semantic Nodes: Object semantic information
    3. Text Embeddings: Object category text representations

    The representation combines these into a unified feature space
    for conditioning the diffusion model.
    """

    def __init__(
        self,
        bev_channels: int = 3,
        semantic_classes: int = 10,
        text_embedding_dim: int = 512,
        hidden_dim: int = 768,
        bev_img_size: int = 64,
        use_clip_text: bool = False,
        bev_model_name: str = "resnet50"
    ):
        super().__init__()

        self.bev_channels = bev_channels
        self.semantic_classes = semantic_classes
        self.hidden_dim = hidden_dim

        # BEV encoder for exploration nodes
        self.bev_encoder = BEVEncoder(
            in_channels=bev_channels,
            hidden_channels=128,
            out_channels=hidden_dim,
            img_size=bev_img_size
        )

        # Semantic encoder for semantic nodes
        self.semantic_encoder = SemanticEncoder(
            num_classes=semantic_classes,
            embedding_dim=hidden_dim,
            img_size=bev_img_size
        )

        # Text embedding layer
        if use_clip_text:
            self.text_encoder = CLIPTextEncoder(
                embedding_dim=hidden_dim,
                freeze=True
            )
        else:
            self.text_encoder = TextEmbeddingLayer(
                num_categories=semantic_classes,
                embedding_dim=hidden_dim,
                use_learnable=True
            )

        # Fusion layer to combine all modalities
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        bev_map: torch.Tensor,
        semantic_map: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        category_names: Optional = None
    ) -> MapRepresentationOutput:
        """
        Args:
            bev_map: (B, bev_channels, H, W) bird's-eye view map
            semantic_map: (B, num_classes, H, W) semantic segmentation
            category_ids: (B, N) category indices for text embedding lookup
            category_names: Optional list of category names for CLIP

        Returns:
            MapRepresentationOutput with all feature tensors
        """
        # Encode exploration features (BEV)
        bev_features = self.bev_encoder(bev_map)

        # Encode semantic features
        semantic_features = self.semantic_encoder(semantic_map)

        # Encode text embeddings
        if category_ids is not None:
            text_features = self.text_encoder(category_ids)
        else:
            # Use zeros if no categories provided
            text_features = torch.zeros(
                bev_map.shape[0], 1, self.hidden_dim,
                device=bev_map.device, dtype=bev_map.dtype
            )

        # Handle text features dimension
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)

        # Pool text features if multiple categories
        text_pooled = text_features.mean(dim=1)  # (B, D)

        # Combine all features
        combined = torch.cat([bev_features, semantic_features, text_pooled], dim=-1)
        combined_features = self.fusion(combined)

        return MapRepresentationOutput(
            bev_features=bev_features,
            semantic_features=semantic_features,
            text_embeddings=text_features,
            combined_features=combined_features
        )


class HierarchicalMapRepresentation(nn.Module):
    """
    Hierarchical map representation with multi-scale features.
    Captures both fine-grained and coarse map information.
    """

    def __init__(
        self,
        bev_channels: int = 3,
        semantic_classes: int = 10,
        hidden_dim: int = 256,
        num_levels: int = 3
    ):
        super().__init__()

        self.num_levels = num_levels
        self.hidden_dim = hidden_dim

        # Multi-scale BEV encoders
        self.bev_encoders = nn.ModuleList([
            BEVEncoder(
                in_channels=bev_channels,
                hidden_channels=hidden_dim // (2 ** i),
                out_channels=hidden_dim,
                img_size=64 // (2 ** i)
            )
            for i in range(num_levels)
        ])

        # Multi-scale semantic encoders
        self.semantic_encoders = nn.ModuleList([
            SemanticEncoder(
                num_classes=semantic_classes,
                embedding_dim=hidden_dim,
                img_size=64 // (2 ** i)
            )
            for i in range(num_levels)
        ])

        # Level fusion
        self.level_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_levels)
        ])

        # Cross-level attention
        self.cross_level_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        bev_map: torch.Tensor,
        semantic_map: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_map: (B, bev_channels, H, W) bird's-eye view
            semantic_map: (B, num_classes, H, W) semantic segmentation

        Returns:
            Dictionary with multi-scale and fused features
        """
        level_features = []

        for i in range(self.num_levels):
            # Downsample for multi-scale
            scale_factor = 2 ** i
            bev_scaled = torch.nn.functional.interpolate(
                bev_map,
                scale_factor=1 / scale_factor,
                mode='bilinear',
                align_corners=False
            )
            semantic_scaled = torch.nn.functional.interpolate(
                semantic_map,
                scale_factor=1 / scale_factor,
                mode='bilinear',
                align_corners=False
            )

            # Encode at this scale
            bev_feat = self.bev_encoders[i](bev_scaled)
            semantic_feat = self.semantic_encoders[i](semantic_scaled)

            # Fuse at level
            level_feat = self.level_fusion[i](
                torch.cat([bev_feat, semantic_feat], dim=-1)
            )
            level_features.append(level_feat)

        # Stack for cross-level attention
        level_stack = torch.stack(level_features, dim=1)  # (B, num_levels, D)

        # Cross-level attention
        attended, _ = self.cross_level_attention(
            level_stack, level_stack, level_stack
        )

        # Final fusion
        attended_flat = attended.flatten(1)  # (B, num_levels * D)
        output = self.final_fusion(attended_flat)

        return {
            'features': output,
            'level_features': level_features,
            'attended': attended
        }
