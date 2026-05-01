"""Main Dual-MapNav diffusion model combining all components."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from models.map_representation import MapRepresentation
from models.observation import ObservationEncoder
from models.diffusion.unet_3d import UNet3D
from models.diffusion.diffusion_process import DiffusionProcess
from models.diffusion.ddim_sampler import DDIMSampler
from models.diffusion.classifier_free_guidance import ClassifierFreeGuidance


@dataclass
class Dual-MapNavOutput:
    """Output container for Dual-MapNav model."""
    generated_maps: torch.Tensor  # (B, C, T, H, W) generated map trajectory
    noise_pred: Optional[torch.Tensor] = None
    map_features: Optional[torch.Tensor] = None
    observation_features: Optional[torch.Tensor] = None


class Dual-MapNavDiffusion(nn.Module):
    """
    Dual-MapNav: Task-Driven Map Learning via Conditional Video Diffusion.

    This model predicts future bird's-eye view maps given:
    - Current RGB-D observation
    - Partial BEV map
    - Text instruction

    Map prediction is formulated as conditional video generation,
    where the video frames are map frames at different timesteps.
    """

    def __init__(
        self,
        # Map representation
        bev_channels: int = 3,
        semantic_classes: int = 10,
        map_embedding_dim: int = 512,

        # Observation encoder
        rgb_backbone: str = "resnet50",
        obs_embed_dim: int = 768,

        # Diffusion model
        diffusion_channels: int = 16,
        unet_base_channels: int = 128,
        unet_channel_multipliers: list = None,
        time_embed_dim: int = 256,
        num_timesteps: int = 1000,

        # Sampling
        guidance_scale: float = 7.5,
        num_sampling_steps: int = 50,

        # Map dimensions
        map_size: int = 64,
        num_map_frames: int = 16,
    ):
        super().__init__()

        if unet_channel_multipliers is None:
            unet_channel_multipliers = [1, 2, 4, 8]

        self.map_size = map_size
        self.num_map_frames = num_map_frames
        self.num_timesteps = num_timesteps
        self.guidance_scale = guidance_scale

        # Map representation encoder
        self.map_encoder = MapRepresentation(
            bev_channels=bev_channels,
            semantic_classes=semantic_classes,
            text_embedding_dim=map_embedding_dim,
            hidden_dim=map_embedding_dim
        )

        # Observation encoder
        self.obs_encoder = ObservationEncoder(
            rgb_backbone=rgb_backbone,
            embed_dim=obs_embed_dim
        )

        # Condition projection (combines map and obs features)
        self.condition_proj = nn.Sequential(
            nn.Linear(map_embedding_dim + obs_embed_dim, obs_embed_dim),
            nn.LayerNorm(obs_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(obs_embed_dim, obs_embed_dim)
        )

        # 3D U-Net for diffusion
        self.unet = UNet3D(
            in_channels=diffusion_channels,
            out_channels=diffusion_channels,
            base_channels=unet_base_channels,
            channel_multipliers=unet_channel_multipliers,
            time_embed_dim=time_embed_dim,
            condition_dim=obs_embed_dim
        )

        # Diffusion process
        self.diffusion = DiffusionProcess(
            num_timesteps=num_timesteps,
            beta_schedule="linear"
        )

        # For map frames, we use fewer channels than RGB
        self.map_channels = diffusion_channels

    def forward(
        self,
        bev_map: torch.Tensor,
        semantic_map: torch.Tensor,
        category_ids: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dual-MapNavOutput:
        """
        Forward pass of Dual-MapNav.

        Args:
            bev_map: (B, bev_channels, H, W) bird's-eye view map
            semantic_map: (B, num_classes, H, W) semantic segmentation
            category_ids: (B, N) category indices for text
            rgb: Optional (B, 3, H, W) RGB observation
            depth: Optional (B, 1, H, W) depth observation
            text_embeddings: Optional pre-computed text embeddings
            timesteps: Optional (B,) for DDPM training
            return_loss: Whether to return training loss

        Returns:
            Dual-MapNavOutput with generated maps or loss
        """
        # Encode map representation
        map_repr = self.map_encoder(bev_map, semantic_map, category_ids)
        map_features = map_repr.combined_features

        # Encode observation
        if rgb is not None or depth is not None:
            obs_features = self.obs_encoder(rgb=rgb, depth=depth)
            if isinstance(obs_features, dict):
                obs_features = obs_features.get('fused', obs_features.get('rgb'))
        else:
            obs_features = torch.zeros_like(map_features)

        # Combine conditions
        condition = self.condition_proj(torch.cat([map_features, obs_features], dim=-1))

        if return_loss and timesteps is not None:
            # Training mode: return loss
            return self._training_forward(condition, bev_map, timesteps)
        else:
            # Inference mode: generate maps
            return self._inference_forward(condition)

    def _training_forward(
        self,
        condition: torch.Tensor,
        target_map: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Dual-MapNavOutput:
        """
        Training forward pass with DDPM loss.
        """
        # Get target map shape
        B, C, H, W = target_map.shape

        # Reshape target for diffusion (add frame dimension)
        # For simplicity, treat each map as a single-frame "video"
        # or use the target as the first frame
        x_start = target_map.unsqueeze(2)  # (B, C, 1, H, W)

        # Repeat to match num_map_frames if needed for 3D conv
        # Here we keep it simple with single frame for map prediction
        target = x_start

        # Sample noise
        noise = torch.randn_like(target)

        # Compute loss
        loss = self.diffusion.training_losses(
            self.unet,
            x_start=target,
            timesteps=timesteps,
            condition=condition,
            noise=noise,
            loss_type="mse"
        )

        # For output, also show what would be generated
        with torch.no_grad():
            # Sample a prediction for visualization
            sampler = DDIMSampler(self.diffusion, self.unet, num_timesteps=min(10, self.num_timesteps))
            generated = sampler.sample(
                condition=condition,
                shape=(self.map_channels, 1, self.map_size, self.map_size),
                device=target.device
            )

        return Dual-MapNavOutput(
            generated_maps=generated,
            noise_pred=None,
            map_features=None,
            observation_features=condition
        )

    def _inference_forward(self, condition: torch.Tensor) -> Dual-MapNavOutput:
        """
        Inference forward pass: generate map from condition.
        """
        # Sample from diffusion
        sampler = DDIMSampler(
            self.diffusion,
            self.unet,
            num_timesteps=self.num_sampling_steps
        )

        generated = sampler.sample(
            condition=condition,
            shape=(self.map_channels, 1, self.map_size, self.map_size),
            device=condition.device
        )

        return Dual-MapNavOutput(
            generated_maps=generated,
            map_features=None,
            observation_features=condition
        )

    @torch.no_grad()
    def generate_map_trajectory(
        self,
        bev_map: torch.Tensor,
        semantic_map: torch.Tensor,
        category_ids: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        num_frames: int = 16,
        use_cfg: bool = True,
        null_condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate a trajectory of maps over time.

        Args:
            bev_map: Initial BEV map
            semantic_map: Initial semantic map
            category_ids: Category indices
            rgb: RGB observation
            depth: Depth observation
            num_frames: Number of frames to generate
            use_cfg: Whether to use classifier-free guidance
            null_condition: Null condition for CFG

        Returns:
            map_trajectory: (B, C, num_frames, H, W) generated map sequence
        """
        self.eval()

        # Encode condition
        map_repr = self.map_encoder(bev_map, semantic_map, category_ids)
        map_features = map_repr.combined_features

        obs_features = self.obs_encoder(rgb=rgb, depth=depth)
        if isinstance(obs_features, dict):
            obs_features = obs_features.get('fused', obs_features.get('rgb'))

        condition = self.condition_proj(torch.cat([map_features, obs_features], dim=-1))

        # Generate frames iteratively
        trajectory = []
        current_map = bev_map

        for frame_idx in range(num_frames):
            # Update condition with temporal information
            frame_condition = condition

            if use_cfg:
                # Use CFG
                if null_condition is None:
                    null_condition = torch.zeros_like(condition)
                sampler = DDIMSampler(self.diffusion, self.unet, num_timesteps=self.num_sampling_steps)
                generated = sampler.sample(
                    condition=frame_condition,
                    null_condition=null_condition,
                    shape=(self.map_channels, 1, self.map_size, self.map_size),
                    device=bev_map.device
                )
            else:
                sampler = DDIMSampler(self.diffusion, self.unet, num_timesteps=self.num_sampling_steps)
                generated = sampler.sample(
                    condition=frame_condition,
                    shape=(self.map_channels, 1, self.map_size, self.map_size),
                    device=bev_map.device
                )

            trajectory.append(generated)

        # Stack frames
        map_trajectory = torch.cat(trajectory, dim=2)  # (B, C, num_frames, H, W)

        return map_trajectory


class Dual-MapNavWithPretrainedDiffusion(nn.Module):
    """
    Dual-MapNav with pretrained video diffusion backbone.
    Uses a pretrained model (e.g., Zeroscope) as the diffusion backbone.
    """

    def __init__(
        self,
        pretrained_model_name: str = "Zeroscope",
        map_channels: int = 16,
        obs_embed_dim: int = 768,
        guidance_scale: float = 7.5,
        map_size: int = 64
    ):
        super().__init__()

        self.map_channels = map_channels
        self.map_size = map_size
        self.guidance_scale = guidance_scale

        # Map encoder
        self.map_encoder = MapRepresentation(
            bev_channels=3,
            semantic_classes=10,
            text_embedding_dim=512,
            hidden_dim=512
        )

        # Observation encoder
        self.obs_encoder = ObservationEncoder(embed_dim=768)

        # Condition adapter
        self.condition_adapter = ConditionAdapter(
            map_dim=512,
            obs_dim=768,
            latent_dim=768
        )

        # For now, use a simple UNet until we load pretrained weights
        self.unet = UNet3D(
            in_channels=map_channels,
            out_channels=map_channels,
            base_channels=128,
            condition_dim=768
        )

        # Diffusion
        self.diffusion = DiffusionProcess()

        # Load pretrained video diffusion if available
        self._load_pretrained(pretrained_model_name)

    def _load_pretrained(self, model_name: str):
        """Load pretrained video diffusion model."""
        # Placeholder for loading pretrained weights
        # In practice, you would use diffusers library:
        # from diffusers import DiffusionPipeline
        # self.pipe = DiffusionPipeline.from_pretrained(model_name)
        pass

    def forward(
        self,
        bev_map: torch.Tensor,
        semantic_map: torch.Tensor,
        category_ids: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dual-MapNavOutput:
        """Forward pass."""
        map_repr = self.map_encoder(bev_map, semantic_map, category_ids)
        obs_features = self.obs_encoder(rgb=rgb, depth=depth)

        condition = self.condition_adapter(map_repr.combined_features, obs_features)

        if timesteps is not None:
            return self._training_forward(condition, bev_map, timesteps)
        else:
            return self._inference_forward(condition)

    def _training_forward(self, condition, target, timesteps):
        noise = torch.randn_like(target)
        loss = self.diffusion.training_losses(self.unet, target, timesteps, condition)
        return Dual-MapNavOutput(generated_maps=None, noise_pred=loss)

    def _inference_forward(self, condition):
        sampler = DDIMSampler(self.diffusion, self.unet)
        generated = sampler.sample(condition, (16, 1, 64, 64))
        return Dual-MapNavOutput(generated_maps=generated)


class ConditionAdapter(nn.Module):
    """
    Adapter to project map and observation features to latent space.
    """

    def __init__(self, map_dim: int, obs_dim: int, latent_dim: int):
        super().__init__()

        self.map_proj = nn.Sequential(
            nn.Linear(map_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True)
        )

        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, map_features: torch.Tensor, obs_features: torch.Tensor) -> torch.Tensor:
        map_proj = self.map_proj(map_features)
        obs_proj = self.obs_proj(obs_features)
        fused = self.fusion(torch.cat([map_proj, obs_proj], dim=-1))
        return fused
