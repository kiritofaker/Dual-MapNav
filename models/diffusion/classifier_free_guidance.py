"""Classifier-Free Guidance for improved conditioning."""

import torch
import torch.nn as nn
from typing import Optional, Callable


class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance implementation.

    Combines conditional and unconditional predictions:
    epsilon = epsilon_uncond + scale * (epsilon_cond - epsilon_uncond)

    This technique allows controlling generation without a classifier.
    """

    def __init__(
        self,
        model: nn.Module,
        guidance_scale: float = 7.5,
        unconditional_tokens: Optional[torch.Tensor] = None
    ):
        """
        Args:
            model: Diffusion model
            guidance_scale: Guidance strength (higher = more adherence to condition)
            unconditional_tokens: Optional tokens for unconditional generation
        """
        self.model = model
        self.guidance_scale = guidance_scale
        self.unconditional_tokens = unconditional_tokens

    @torch.no_grad()
    def sample(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        null_condition: torch.Tensor,
        noise_pred_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Perform classifier-free guided sampling step.

        Args:
            x_t: (B, C, T, H, W) current noisy sample
            timesteps: (B,) timestep
            condition: (B, D) conditioning
            null_condition: (B, D) null conditioning
            noise_pred_fn: Optional custom noise prediction function

        Returns:
            guided_noise_pred: (B, C, T, H, W) guided noise prediction
        """
        # Conditional prediction
        if noise_pred_fn is not None:
            noise_pred_cond = noise_pred_fn(x_t, timesteps, condition)
            noise_pred_uncond = noise_pred_fn(x_t, timesteps, null_condition)
        else:
            noise_pred_cond = self.model(x_t, timesteps, condition)
            noise_pred_uncond = self.model(x_t, timesteps, null_condition)

        # CFG formula
        guided = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        return guided

    def compute_guidance_weight(self, timesteps: torch.Tensor, mode: str = "constant") -> torch.Tensor:
        """
        Compute time-dependent guidance weight.

        Args:
            timesteps: (B,) timesteps
            mode: Weighting scheme ("constant", "linear", "quadratic")

        Returns:
            weights: (B,) guidance weights
        """
        if mode == "constant":
            weights = torch.ones_like(timesteps.float())
        elif mode == "linear":
            weights = timesteps.float() / timesteps.max()
        elif mode == "quadratic":
            weights = (timesteps.float() / timesteps.max()) ** 2
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return weights


class AdaptiveClassifierFreeGuidance:
    """
    Adaptive CFG with guidance scale scheduling.
    Adjusts guidance strength based on timestep.
    """

    def __init__(
        self,
        model: nn.Module,
        base_guidance_scale: float = 7.5,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 15.0,
        guidance_schedule: str = "linear"
    ):
        self.model = model
        self.base_guidance_scale = base_guidance_scale
        self.min_guidance_scale = min_guidance_scale
        self.max_guidance_scale = max_guidance_scale
        self.guidance_schedule = guidance_schedule

    def get_guidance_scale(self, timestep: int, num_timesteps: int) -> float:
        """
        Compute guidance scale for current timestep.

        Args:
            timestep: Current timestep
            num_timesteps: Total number of timesteps

        Returns:
            guidance_scale: Scaled guidance value
        """
        progress = timestep / num_timesteps

        if self.guidance_schedule == "linear":
            # Higher guidance early, lower later
            scale = self.max_guidance_scale - (self.max_guidance_scale - self.min_guidance_scale) * progress
        elif self.guidance_schedule == "constant":
            scale = self.base_guidance_scale
        elif self.guidance_schedule == "step":
            # Drop guidance after certain point
            if progress > 0.5:
                scale = self.min_guidance_scale
            else:
                scale = self.max_guidance_scale
        else:
            scale = self.base_guidance_scale

        return scale

    @torch.no_grad()
    def sample(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        null_condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform adaptive CFG sampling step.
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Compute guidance for each sample in batch
        guidance_scales = torch.tensor([
            self.get_guidance_scale(t.item(), self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 1000)
            for t in timesteps
        ], device=device)

        # Conditional prediction
        noise_pred_cond = self.model(x_t, timesteps, condition)
        noise_pred_uncond = self.model(x_t, timesteps, null_condition)

        # Apply adaptive guidance
        guided = noise_pred_uncond + guidance_scales.view(-1, 1, 1, 1, 1) * (noise_pred_cond - noise_pred_uncond)

        return guided


class DropoutCFG:
    """
    Classifier-Free Guidance via random dropout.
    Uses dropout at inference time to approximate CFG.
    """

    def __init__(
        self,
        model: nn.Module,
        guidance_scale: float = 7.5,
        dropout_prob: float = 0.1
    ):
        self.model = model
        self.guidance_scale = guidance_scale
        self.dropout_prob = dropout_prob
        self.original_training = model.training

    @torch.no_grad()
    def sample(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        use_dropout: bool = True
    ) -> torch.Tensor:
        """
        Sample with dropout-based CFG approximation.
        """
        # Enable dropout for unconditional approximation
        if use_dropout and self.dropout_prob > 0:
            self.model.train()

        # First pass with condition
        noise_pred_cond = self.model(x_t, timesteps, condition)

        # Second pass with zeroed condition (approximates unconditional)
        null_condition = torch.zeros_like(condition)
        noise_pred_uncond = self.model(x_t, timesteps, null_condition)

        # Restore training state
        if not self.original_training:
            self.model.eval()

        # CFG
        guided = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        return guided

    def __enter__(self):
        self.original_training = self.model.training
        return self

    def __exit__(self, *args):
        if not self.original_training:
            self.model.eval()
