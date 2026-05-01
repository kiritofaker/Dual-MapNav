"""Text Embedding Layer for object category representations."""

import torch
import torch.nn as nn
from typing import List, Optional


class TextEmbeddingLayer(nn.Module):
    """
    Encodes object category names into embeddings.

    Supports:
    - Direct embedding lookup
    - CLIP-encoded text features (when available)
    - Learnable category embeddings
    """

    def __init__(
        self,
        num_categories: int = 100,
        embedding_dim: int = 512,
        use_learnable: bool = True,
        clip_model_name: str = "ViT-L/14"
    ):
        super().__init__()

        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.use_learnable = use_learnable

        if use_learnable:
            # Learnable category embeddings
            self.category_embeddings = nn.Parameter(
                torch.randn(num_categories, embedding_dim),
                requires_grad=True
            )
            nn.init.normal_(self.category_embeddings, mean=0, std=0.02)
        else:
            # Fixed embeddings (e.g., CLIP text encoder output)
            self.category_embeddings = None

        # Optional projection for CLIP features
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, category_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            category_ids: (B, N) or (B,) category indices

        Returns:
            text_features: (B, D) or (B, N, D) encoded text features
        """
        if self.category_embeddings is not None:
            embeddings = self.category_embeddings[category_ids]
        else:
            embeddings = category_ids

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        return embeddings


class CLIPTextEncoder(nn.Module):
    """
    Uses CLIP text encoder for high-quality text embeddings.
    Provides better vision-language alignment for navigation.
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        embedding_dim: int = 768,
        freeze: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim

        try:
            import clip
            self.clip_model, _ = clip.load(model_name, device='cpu')
            self.clip_model.eval()

            if freeze:
                for param in self.clip_model.parameters():
                    param.requires_grad = False

            # Text projection
            self.projection = nn.Linear(768, embedding_dim)

        except ImportError:
            print("Warning: CLIP not available. Install with: pip install clip-by-openai")
            self.clip_model = None
            self.projection = nn.Linear(512, embedding_dim)

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode text prompts using CLIP.

        Args:
            text: List of text strings

        Returns:
            text_features: (B, D) normalized text features
        """
        if self.clip_model is not None:
            import clip
            text_tokens = clip.tokenize(text).to(next(self.clip_model.parameters()).device)
            with torch.no_grad():
                features = self.clip_model.encode_text(text_tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        else:
            # Fallback to random features
            features = torch.randn(len(text), 512)

        features = self.projection(features)
        return features

    def forward(
        self,
        category_ids: torch.Tensor,
        category_names: Optional[List[List[str]]] = None
    ) -> torch.Tensor:
        """
        Args:
            category_ids: (B, N) category indices
            category_names: Optional list of category names for CLIP encoding

        Returns:
            text_features: (B, N, D) text embeddings for each category
        """
        B, N = category_ids.shape

        if category_names is not None and self.clip_model is not None:
            # Encode with CLIP
            all_features = []
            for batch_names in category_names:
                batch_features = self.encode_text(batch_names)
                all_features.append(batch_features)
            features = torch.stack(all_features)
        else:
            # Use learnable embeddings
            features = self.category_embeddings[category_ids]

        return features


class InstructionEncoder(nn.Module):
    """
    Encodes navigation instructions (e.g., "Go to the kitchen").
    Used for conditioning the diffusion model.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_length, embedding_dim),
            requires_grad=False
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token indices

        Returns:
            instruction_features: (B, D) encoded instruction
        """
        B, L = input_ids.shape

        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids)
        x = x + self.pos_encoding[:, :L, :]

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output projection
        output = self.output_proj(x)

        return output
