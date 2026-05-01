"""RGB Encoder for observation processing."""

import torch
import torch.nn as nn
from typing import Optional


class RGBEncoder(nn.Module):
    """
    Encodes RGB images from robot observations.
    Uses a pretrained backbone (ResNet or ViT).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 2048,
        freeze_bn: bool = True
    ):
        super().__init__()

        self.backbone_name = backbone
        self.output_dim = output_dim

        if backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone_model = resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(backbone_model.children())[:-2])
            self.feature_dim = 2048
        elif backbone == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V2 if pretrained else None
            backbone_model = resnet34(weights=weights)
            self.encoder = nn.Sequential(*list(backbone_model.children())[:-2])
            self.feature_dim = 512
        elif backbone == "vit":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            backbone_model = vit_b_16(weights=weights)
            self.encoder = backbone_model
            self.feature_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Output projection
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) if not backbone.startswith("vit") else nn.Identity(),
            nn.Flatten(),
            nn.Linear(self.feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) RGB image

        Returns:
            features: (B, D) encoded RGB features
        """
        if self.backbone_name.startswith("vit"):
            features = self.encoder(rgb)
        else:
            features = self.encoder(rgb)
            features = self.projection(features)

        return features


class CLIPVisionEncoder(nn.Module):
    """
    CLIP-based vision encoder for better vision-language alignment.
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        output_dim: int = 768,
        freeze: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.output_dim = output_dim

        try:
            import clip
            self.clip_model, _ = clip.load(model_name, device='cpu')
            self.clip_model.eval()

            if freeze:
                for param in self.clip_model.parameters():
                    param.requires_grad = False

            self.feature_dim = 1024 if "L" in model_name else 512

        except ImportError:
            print("Warning: CLIP not available")
            self.clip_model = None
            self.feature_dim = 1024

        # Output projection
        self.projection = nn.Linear(self.feature_dim, output_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) RGB image

        Returns:
            features: (B, D) CLIP vision features
        """
        if self.clip_model is not None:
            import clip
            features = self.clip_model.encode_image(rgb)
            features = features / features.norm(dim=-1, keepdim=True)
        else:
            features = torch.randn(rgb.shape[0], self.feature_dim, device=rgb.device)

        features = self.projection(features)
        return features
