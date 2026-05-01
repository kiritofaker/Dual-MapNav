"""DDIM Sampler for faster sampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import numpy as np


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.

    Provides faster sampling with deterministic trajectories.
    """

    def __init__(
        self,
        diffusion: nn.Module,
        model: nn.Module,
        num_timesteps: int = 50,
        eta: float = 0.0,
        clip_range: tuple = (-1.0, 1.0)
    ):
        """
        Args:
            diffusion: DiffusionProcess instance
            model: Neural network model
            num_timesteps: Number of sampling steps (typically 20-100)
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM-like)
            clip_range: Range for clipping predictions
        """
        self.diffusion = diffusion
        self.model = model
        self.num_timesteps = num_timesteps
        self.eta = eta
        self.clip_range = clip_range

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        shape: tuple,
        device: torch.device = None,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples using DDIM.

        Args:
            condition: (B, D) conditioning tensor
            shape: (C, T, H, W) desired output shape
            device: Target device
            seed: Optional random seed

        Returns:
            x_0: (B, C, T, H, W) generated map sequence
        """
        if seed is not None:
            torch.manual_seed(seed)

        batch_size = condition.shape[0]
        if device is None:
            device = condition.device

        # Initialize from noise
        x_t = torch.randn(batch_size, *shape, device=device)

        # Create timestep sequence (from T to 0)
        timesteps = torch.linspace(
            self.diffusion.num_timesteps - 1,
            0,
            self.num_timesteps,
            dtype=torch.long,
            device=device
        )

        # Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(x_t, t_batch, condition)

            # Predict x_0
            x_0_pred = self.diffusion.predict_start_from_noise(x_t, t_batch, noise_pred)
            x_0_pred = x_0_pred.clamp(self.clip_range[0], self.clip_range[1])

            # Get previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = torch.zeros_like(t)

            # DDIM update step
            x_t = self._ddim_step(x_t, x_0_pred, t, t_prev)

        return x_0_pred

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a single DDIM step.

        DDIM update rule:
        x_{t-1} = sqrt(alpha_{t-1}) * x_0_pred + sqrt(1 - alpha_{t-1}) * noise_pred

        where noise_pred is derived from x_t and x_0_pred.

        Args:
            x_t: (B, C, T, H, W) current noisy sample
            x_0_pred: (B, C, T, H, W) predicted clean sample
            t: Current timestep
            t_prev: Previous timestep

        Returns:
            x_{t-1}: Previous timestep sample
        """
        alpha_t = self.diffusion.alphas_cumprod[t]
        alpha_t_prev = self.diffusion.alphas_cumprod[t_prev] if t_prev > 0 else torch.ones_like(alpha_t)

        # Predict direction to x_t
        pred_original_sample = x_0_pred
        pred_sample_direction = (1 - alpha_t).sqrt() * (
            (x_t - alpha_t.sqrt() * pred_original_sample) / (1 - alpha_t).sqrt()
        )

        # Compute x_{t-1}
        x_prev = alpha_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        # Add stochasticity if eta > 0
        if self.eta > 0:
            variance = self._get_variance(t, t_prev)
            noise = torch.randn_like(x_t)
            x_prev = x_prev + variance.sqrt() * noise

        return x_prev

    def _get_variance(self, t: torch.Tensor, t_prev: torch.Tensor) -> torch.Tensor:
        """Compute variance for stochastic term."""
        alpha_t = self.diffusion.alphas_cumprod[t]
        alpha_t_prev = self.diffusion.alphas_cumprod[t_prev] if t_prev > 0 else torch.ones_like(alpha_t)

        beta_t = 1 - alpha_t / alpha_t_prev
        variance = (1 - alpha_t_prev) / (1 - alpha_t) * beta_t

        # Scale by eta
        variance = self.eta * variance

        return variance


class DDIMSamplerWithCFG(DDIMSampler):
    """
    DDIM sampler with Classifier-Free Guidance.
    """

    def __init__(
        self,
        diffusion: nn.Module,
        model: nn.Module,
        num_timesteps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        clip_range: tuple = (-1.0, 1.0)
    ):
        super().__init__(diffusion, model, num_timesteps, eta, clip_range)
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        null_condition: torch.Tensor,
        shape: tuple,
        device: torch.device = None,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples with classifier-free guidance.

        Args:
            condition: (B, D) conditioning tensor
            null_condition: (B, D) null/unconditional tensor
            shape: (C, T, H, W) desired output shape
            device: Target device
            seed: Optional random seed

        Returns:
            x_0: (B, C, T, H, W) generated map sequence
        """
        if seed is not None:
            torch.manual_seed(seed)

        batch_size = condition.shape[0]
        if device is None:
            device = condition.device

        # Initialize from noise
        x_t = torch.randn(batch_size, *shape, device=device)

        # Create timestep sequence
        timesteps = torch.linspace(
            self.diffusion.num_timesteps - 1,
            0,
            self.num_timesteps,
            dtype=torch.long,
            device=device
        )

        # Denoising loop with CFG
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Conditional prediction
            noise_pred_cond = self.model(x_t, t_batch, condition)

            # Unconditional prediction
            noise_pred_uncond = self.model(x_t, t_batch, null_condition)

            # Classifier-free guidance
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Predict x_0
            x_0_pred = self.diffusion.predict_start_from_noise(x_t, t_batch, noise_pred)
            x_0_pred = x_0_pred.clamp(self.clip_range[0], self.clip_range[1])

            # Get previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = torch.zeros_like(t)

            # DDIM update
            x_t = self._ddim_step(x_t, x_0_pred, t, t_prev)

        return x_0_pred
