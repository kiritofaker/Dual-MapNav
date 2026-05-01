"""Semantic Encoder for semantic map nodes."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticEncoder(nn.Module):
    """
    Encodes semantic segmentation maps containing object semantic information.

    The semantic map represents:
    - Object class segmentation
    - Instance boundaries
    - Object depth ordering
    """

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 512,
        img_size: int = 64
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Class embeddings for semantic categories
        self.class_embeddings = nn.Parameter(
            torch.randn(num_classes, embedding_dim // 4),
            requires_grad=True
        )

        # Convolutional encoder
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(num_classes + 3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Global pooling and projection
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 16, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, semantic_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_map: (B, num_classes, H, W) one-hot or soft semantic segmentation

        Returns:
            semantic_features: (B, D) encoded semantic features
        """
        x = self.encoder(semantic_map)
        x = self.projection(x)
        return x


class SemanticQueryEncoder(nn.Module):
    """
    Transformer-based encoder that treats semantic map as a set of object queries.
    More flexible for handling variable number of objects.
    """

    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 4
    ):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Class query embeddings (learnable)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Input projection for semantic features
        self.input_proj = nn.Sequential(
            nn.Conv2d(num_classes, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Transformer encoder for object queries
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        semantic_map: torch.Tensor,
        object_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            semantic_map: (B, num_classes, H, W) semantic segmentation
            object_mask: (B, H, W) optional mask for valid regions

        Returns:
            query_features: (B, num_queries, D) object query features
        """
        B, C, H, W = semantic_map.shape

        # Project to hidden dimension
        x = self.input_proj(semantic_map)  # (B, hidden_dim, H, W)

        # Reshape to sequence
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)

        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: queries attend to semantic features
        # Use self-attention on queries for interaction
        query_features = self.transformer(query_embed)

        # Output projection
        output = self.output_proj(query_features)  # (B, num_queries, hidden_dim)

        return output
