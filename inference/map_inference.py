"""Inference module for map generation."""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import numpy as np

from models.map_predictor import Dual-MapNavDiffusion
from models.diffusion.ddim_sampler import DDIMSampler


class MapInference:
    """
    Inference wrapper for Dual-MapNav map generation.
    Handles sampling and trajectory generation.
    """

    def __init__(
        self,
        model: Dual-MapNavDiffusion,
        device: str = "cuda",
        num_sampling_steps: int = 50,
        guidance_scale: float = 7.5
    ):
        self.model = model
        self.device = device
        self.num_sampling_steps = num_sampling_steps
        self.guidance_scale = guidance_scale

        # Move model to device and eval mode
        self.model.to(device)
        self.model.eval()

        # Create sampler
        self.sampler = DDIMSampler(
            diffusion=self.model.diffusion,
            model=self.model.unet,
            num_timesteps=num_sampling_steps,
            guidance_scale=guidance_scale
        )

    @torch.no_grad()
    def generate_map(
        self,
        bev_map: torch.Tensor,
        semantic_map: torch.Tensor,
        category_ids: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        use_cfg: bool = True,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate a single map prediction.

        Args:
            bev_map: (B, 3, H, W) bird's-eye view
            semantic_map: (B, num_classes, H, W) semantic segmentation
            category_ids: (B, N) category indices
            rgb: Optional (B, 3, H, W) RGB observation
            depth: Optional (B, 1, H, W) depth observation
            use_cfg: Whether to use classifier-free guidance
            seed: Random seed

        Returns:
            generated_map: (B, C, T, H, W) generated map
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Move inputs to device
        bev_map = bev_map.to(self.device)
        semantic_map = semantic_map.to(self.device)
        category_ids = category_ids.to(self.device)

        if rgb is not None:
            rgb = rgb.to(self.device)
        if depth is not None:
            depth = depth.to(self.device)

        # Encode conditions
        map_repr = self.model.map_encoder(bev_map, semantic_map, category_ids)
        obs_features = self.model.obs_encoder(rgb=rgb, depth=depth)

        if isinstance(obs_features, dict):
            obs_features = obs_features.get('fused', obs_features.get('rgb'))

        condition = self.model.condition_proj(
            torch.cat([map_repr.combined_features, obs_features], dim=-1)
        )

        # Sample
        if use_cfg:
            null_condition = torch.zeros_like(condition)
            generated = self.sampler.sample(
                condition=condition,
                null_condition=null_condition,
                shape=(self.model.map_channels, 1, self.model.map_size, self.model.map_size),
                device=self.device
            )
        else:
            generated = self.sampler.sample(
                condition=condition,
                shape=(self.model.map_channels, 1, self.model.map_size, self.model.map_size),
                device=self.device
            )

        return generated

    @torch.no_grad()
    def generate_trajectory(
        self,
        bev_map: torch.Tensor,
        semantic_map: torch.Tensor,
        category_ids: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        num_frames: int = 16,
        use_cfg: bool = True,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate a trajectory of maps (video-like output).

        Args:
            bev_map: (B, 3, H, W) initial bird's-eye view
            semantic_map: (B, num_classes, H, W) initial semantic map
            category_ids: (B, N) category indices
            rgb: Optional RGB observation
            depth: Optional depth observation
            num_frames: Number of frames to generate
            use_cfg: Whether to use CFG
            seed: Random seed

        Returns:
            trajectory: (B, C, num_frames, H, W) map trajectory
        """
        if seed is not None:
            torch.manual_seed(seed)

        trajectory = []

        # Use initial map as starting point
        current_bev = bev_map.clone()
        current_semantic = semantic_map.clone()

        for frame_idx in range(num_frames):
            # Generate next map
            generated = self.generate_map(
                bev_map=current_bev,
                semantic_map=current_semantic,
                category_ids=category_ids,
                rgb=rgb,
                depth=depth,
                use_cfg=use_cfg
            )

            trajectory.append(generated)

            # Update for next frame (simple autoregressive)
            # In practice, you'd update based on predicted changes
            if frame_idx < num_frames - 1:
                current_bev = self._update_map(current_bev, generated, frame_idx / num_frames)

        # Stack frames
        trajectory = torch.cat(trajectory, dim=2)  # (B, C, num_frames, H, W)

        return trajectory

    def _update_map(
        self,
        current: torch.Tensor,
        generated: torch.Tensor,
        progress: float
    ) -> torch.Tensor:
        """Update current map based on generated prediction."""
        # Blend current and generated based on progress
        alpha = min(1.0, progress * 2)  # Gradually increase influence
        updated = (1 - alpha) * current + alpha * generated.squeeze(2)

        # Ensure 3 channels for BEV
        if updated.shape[1] > 3:
            updated = updated[:, :3, :, :]

        return updated

    def batch_generate(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int = 1
    ) -> List[torch.Tensor]:
        """
        Generate multiple samples for a single input.

        Args:
            batch: Dictionary with input tensors
            num_samples: Number of samples to generate

        Returns:
            List of generated maps
        """
        results = []

        for i in range(num_samples):
            seed = i * 42 if i is not None else None
            generated = self.generate_map(
                bev_map=batch['bev_map'],
                semantic_map=batch['semantic_map'],
                category_ids=batch['category_ids'],
                rgb=batch.get('rgb'),
                depth=batch.get('depth'),
                seed=seed
            )
            results.append(generated)

        return results


def visualize_map(
    bev_map: torch.Tensor,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Convert BEV map tensor to numpy for visualization.

    Args:
        bev_map: (C, H, W) or (B, C, H, W) tensor
        save_path: Optional path to save visualization

    Returns:
        rgb_img: (H, W, 3) numpy array
    """
    if bev_map.dim() == 4:
        bev_map = bev_map[0]  # Take first batch

    bev_np = bev_map.cpu().numpy()

    # Channel 0: traversability (grayscale)
    traversable = bev_np[0] if bev_np.shape[0] >= 1 else np.zeros_like(bev_np[0])

    # Create RGB visualization
    rgb = np.zeros((traversable.shape[0], traversable.shape[1], 3))

    # Traversability as green
    rgb[:, :, 1] = traversable

    # Add distance as blue tint
    if bev_np.shape[0] >= 2:
        distance = bev_np[1]
        rgb[:, :, 2] = distance * 0.5

    # Add gradient as red
    if bev_np.shape[0] >= 3:
        gradient = bev_np[2]
        rgb[:, :, 0] = gradient * 0.5

    # Normalize to 0-255
    rgb = (rgb * 255).astype(np.uint8)
    rgb = np.clip(rgb, 0, 255)

    if save_path is not None:
        import cv2
        cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return rgb
