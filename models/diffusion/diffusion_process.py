"""Diffusion process for map trajectory generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import numpy as np


def get_beta_schedule(
    schedule_name: str = "linear",
    num_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> np.ndarray:
    """
    Get beta schedule for diffusion.

    Args:
        schedule_name: Type of schedule (linear, cosine, quadratic)
        num_timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        betas: (num_timesteps,) array of beta values
    """
    if schedule_name == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps)

    elif schedule_name == "cosine":
        # Cosine schedule as in ADM
        steps = num_timesteps + 1
        x = np.linspace(0, num_timesteps, steps)
        alphas_cumprod = np.cos(((x / num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)

    elif schedule_name == "quadratic":
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2

    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    return betas.astype(np.float32)


class DiffusionProcess(nn.Module):
    """
    DDPM diffusion process for map trajectory generation.

    Forward process: q(x_t | x_0) = N(sqrt(1 - beta_t) * x_0, beta_t * I)
    Reverse process: p_theta(x_{t-1} | x_t, c) predicted by neural network
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_range: tuple = (-1.0, 1.0)
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.clip_range = clip_range

        # Beta schedule
        betas = get_beta_schedule(
            schedule_name=beta_schedule,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )
        self.register_buffer('betas', torch.from_numpy(betas))

        # Precompute diffusion constants
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])

        self.register_buffer('alphas', torch.from_numpy(alphas))
        self.register_buffer('alphas_cumprod', torch.from_numpy(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', torch.from_numpy(alphas_cumprod_prev))

        # For q(x_t | x_0, x_t-1)
        self.register_buffer('sqrt_alphas_cumprod', torch.from_numpy(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.from_numpy(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas', torch.from_numpy(np.sqrt(1.0 / alphas)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.from_numpy(np.sqrt(1 / alphas_cumprod - 1)))

        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', torch.from_numpy(posterior_variance))

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean sample.

        Args:
            x_start: (B, C, T, H, W) clean map sequence
            timesteps: (B,) timestep indices
            noise: Optional pre-sampled noise

        Returns:
            x_t: (B, C, T, H, W) noisy sample
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor
    ) -> tuple:
        """
        Compute mean and variance of posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: (B, C, T, H, W) clean sample
            x_t: (B, C, T, H, W) noisy sample at time t
            timesteps: (B,) timestep indices

        Returns:
            posterior_mean, posterior_variance
        """
        posterior_mean = (
            self._extract(self.posterior_variance, timesteps, x_t.shape) ** 0.5 *
            self._extract(self.sqrt_recip_alphas, timesteps, x_t.shape) * x_t +
            self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * x_start
        )
        posterior_variance = self._extract(self.posterior_variance, timesteps, x_t.shape)

        return posterior_mean, posterior_variance

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from noisy sample and predicted noise.

        Args:
            x_t: (B, C, T, H, W) noisy sample
            timesteps: (B,) timestep indices
            noise: (B, C, T, H, W) predicted noise

        Returns:
            x_pred: (B, C, T, H, W) predicted clean sample
        """
        x_pred = (
            self._extract(self.sqrt_recip_alphas, timesteps, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )
        return x_pred

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        clip_range: tuple = None
    ) -> torch.Tensor:
        """
        Single reverse diffusion step (DDPM sampling).

        Args:
            model: Diffusion model
            x_t: (B, C, T, H, W) current noisy sample
            timesteps: (B,) current timestep
            condition: (B, D) conditioning
            clip_range: Optional range for clipping predictions

        Returns:
            x_{t-1}: Previous timestep sample
        """
        # Predict noise
        noise_pred = model(x_t, timesteps, condition)

        # Get predicted x_0
        x_pred = self.predict_start_from_noise(x_t, timesteps, noise_pred)

        # Clip
        if clip_range is None:
            clip_range = self.clip_range
        x_pred = x_pred.clamp(clip_range[0], clip_range[1])

        # Compute posterior
        posterior_mean, _ = self.q_posterior_mean_variance(x_pred, x_t, timesteps)

        # For t > 0, add noise
        if timesteps.min() > 0:
            noise = torch.randn_like(x_t)
            # Don't add noise on final step
            noise = noise * (timesteps > 0).float().view(-1, 1, 1, 1, 1)
            x_prev = posterior_mean + torch.sqrt(self.posterior_variance[timesteps]) * noise
        else:
            x_prev = posterior_mean

        return x_prev

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        loss_type: str = "mse"
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            model: Diffusion model
            x_start: (B, C, T, H, W) clean map sequence
            timesteps: (B,) random timesteps
            condition: (B, D) conditioning
            noise: Optional pre-sampled noise
            loss_type: Type of loss (mse, l1, huber)

        Returns:
            loss: Training loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Add noise
        x_t = self.q_sample(x_start, timesteps, noise)

        # Predict noise
        noise_pred = model(x_t, timesteps, condition)

        # Compute loss
        if loss_type == "mse":
            loss = F.mse_loss(noise_pred, noise, reduction='none')
        elif loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise, reduction='none')
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise_pred, noise, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Mean over all dimensions, sum over batch
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        return loss.sum()

    def _extract(
        self,
        coefficients: torch.Tensor,
        timesteps: torch.Tensor,
        target_shape: tuple
    ) -> torch.Tensor:
        """
        Extract coefficients at specific timesteps and reshape for broadcasting.

        Args:
            coefficients: (num_timesteps,) coefficient array
            timesteps: (B,) timestep indices
            target_shape: Shape to broadcast to

        Returns:
            extracted: Broadcasted coefficients
        """
        batch_size = timesteps.shape[0]
        out = coefficients.to(timesteps.device).gather(0, timesteps)
        return out.view(batch_size, *([1] * (len(target_shape) - 1)))


class LatentDiffusionProcess(DiffusionProcess):
    """
    Latent diffusion for more efficient training.
    Operates in a compressed latent space.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        latent_scale_factor: float = 0.18215
    ):
        super().__init__(num_timesteps, beta_schedule)
        self.latent_scale_factor = latent_scale_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        return x / self.latent_scale_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return z * self.latent_scale_factor

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        loss_type: str = "mse"
    ) -> torch.Tensor:
        """Training loss in latent space."""
        # Encode to latent
        z_start = self.encode(x_start)
        return super().training_losses(model, z_start, timesteps, condition, noise, loss_type)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q in latent space."""
        z_start = self.encode(x_start)
        return super().q_sample(z_start, timesteps, noise)

    def decode_sample(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image space."""
        return self.decode(z)
