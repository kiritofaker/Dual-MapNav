"""Cross-Modal Fusion module for combining RGB, Depth, and BEV observations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class CrossModalFusion(nn.Module):
    """
    Cross-modal attention-based fusion for RGB, Depth, and BEV features.
    Uses self-attention and cross-attention to combine modalities.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 4
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Input projection for each modality
        self.rgb_proj = nn.Linear(embed_dim, embed_dim)
        self.depth_proj = nn.Linear(embed_dim, embed_dim)
        self.bev_proj = nn.Linear(embed_dim, embed_dim)

        # Positional encoding for each modality
        self.rgb_pos = PositionalEncoding(embed_dim)
        self.depth_pos = PositionalEncoding(embed_dim)
        self.bev_pos = PositionalEncoding(embed_dim)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Self-attention for fused features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
        bev_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb_features: (B, D) or (B, seq_len, D) RGB features
            depth_features: (B, D) or (B, seq_len, D) Depth features
            bev_features: (B, D) or (B, seq_len, D) BEV features

        Returns:
            fused_rgb, fused_depth, fused_bev: Cross-modal fused features
        """
        # Handle 2D input (B, D) -> (B, 1, D)
        if rgb_features.dim() == 2:
            rgb_features = rgb_features.unsqueeze(1)
            depth_features = depth_features.unsqueeze(1)
            bev_features = bev_features.unsqueeze(1)

        # Project to common space
        rgb = self.rgb_proj(rgb_features)
        depth = self.depth_proj(depth_features)
        bev = self.bev_proj(bev_features)

        # Apply positional encoding
        rgb = self.rgb_pos(rgb)
        depth = self.depth_pos(depth)
        bev = self.bev_pos(bev)

        # Stack as sequence: [RGB, Depth, BEV]
        # Each modality attends to others via cross-attention
        for layer in self.cross_attention_layers:
            rgb, depth, bev = layer(rgb, depth, bev)

        # Global fusion via self-attention
        fused = torch.cat([rgb, depth, bev], dim=1)
        attended, _ = self.self_attention(fused, fused, fused)

        # Pool each modality
        fused_rgb = attended[:, :rgb.shape[1]].mean(dim=1)
        fused_depth = attended[:, rgb.shape[1]:rgb.shape[1]+depth.shape[1]].mean(dim=1)
        fused_bev = attended[:, -bev.shape[1]:].mean(dim=1)

        # Output projection
        output = self.output_proj(
            torch.cat([fused_rgb, fused_depth, fused_bev], dim=-1)
        )

        return fused_rgb, fused_depth, fused_bev, output


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer for modality interaction."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # Query, Key, Value projections for each modality
        self.qkv_proj = nn.ModuleDict({
            'rgb': nn.Linear(embed_dim, embed_dim * 3),
            'depth': nn.Linear(embed_dim, embed_dim * 3),
            'bev': nn.Linear(embed_dim, embed_dim * 3),
        })

        # Cross-attention: each modality attends to others
        self.cross_attn_rgb = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.cross_attn_depth = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.cross_attn_bev = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        bev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: (B, L_rgb, D) RGB features
            depth: (B, L_depth, D) Depth features
            bev: (B, L_bev, D) BEV features

        Returns:
            rgb, depth, bev: Updated features after cross-attention
        """
        # Cross-attention: RGB attends to Depth and BEV
        rgb_qkv = self.qkv_proj['rgb'](rgb).chunk(3, dim=-1)
        depth_kv = self.qkv_proj['depth'](depth).chunk(3, dim=-1)
        bev_kv = self.qkv_proj['bev'](bev).chunk(3, dim=-1)

        # RGB attends to Depth
        rgb_attended, _ = self.cross_attn_rgb(
            query=rgb_qkv[0],
            key=torch.cat([depth_kv[0], bev_kv[0]], dim=1),
            value=torch.cat([depth_kv[2], bev_kv[2]], dim=1)
        )
        rgb = self.norm1(rgb + rgb_attended)

        # Depth attends to RGB and BEV
        depth_attended, _ = self.cross_attn_depth(
            query=depth_kv[0],
            key=torch.cat([rgb_qkv[0], bev_kv[0]], dim=1),
            value=torch.cat([rgb_qkv[2], bev_kv[2]], dim=1)
        )
        depth = self.norm2(depth + depth_attended)

        # BEV attends to RGB and Depth
        bev_attended, _ = self.cross_attn_bev(
            query=bev_kv[0],
            key=torch.cat([rgb_qkv[0], depth_kv[0]], dim=1),
            value=torch.cat([rgb_qkv[2], depth_kv[2]], dim=1)
        )
        bev = self.norm3(bev + bev_attended)

        # FFN
        rgb = self.ffn(rgb)
        depth = self.ffn(depth)
        bev = self.ffn(bev)

        return rgb, depth, bev


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input features

        Returns:
            x with positional encoding added
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]


class AttentionFusion(nn.Module):
    """
    Simple attention-based fusion without cross-attention layers.
    More memory-efficient for larger inputs.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1)
        )

        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        features_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            features_list: List of (B, D) feature tensors

        Returns:
            fused: (B, D) attended fusion
        """
        stacked = torch.stack(features_list, dim=1)  # (B, N, D)
        attn_weights = self.attention(stacked)  # (B, N, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        fused = (stacked * attn_weights).sum(dim=1)  # (B, D)
        fused = self.projection(fused)

        return fused
