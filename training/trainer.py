"""Training pipeline for Dual-MapNav."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import os
import json
from tqdm import tqdm
import wandb

from models.map_predictor import Dual-MapNavDiffusion
from data.synthetic_data import SyntheticMapDataset


class Trainer:
    """
    Base trainer for Dual-MapNav.
    Handles training loop, logging, and checkpointing.
    """

    def __init__(
        self,
        model: Dual-MapNavDiffusion,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100)
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Logging
        self.log_frequency = config.get('log_frequency', 100)
        self.use_wandb = config.get('use_wandb', False)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        num_epochs = self.config.get('max_epochs', 100)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch)
            else:
                val_loss = None

            # Update scheduler
            self.scheduler.step()

            # Log
            log_dict = {'epoch': epoch, 'train_loss': train_loss}
            if val_loss is not None:
                log_dict['val_loss'] = val_loss
            log_dict['lr'] = self.scheduler.get_last_lr()[0]

            self._log(log_dict)

            # Save checkpoint
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')

            # Save best
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.save_checkpoint('best.pt')

        print(f"Training complete. Best loss: {self.best_loss:.4f}")

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()

            output = self.model(
                bev_map=batch['bev_map'],
                semantic_map=batch['semantic_map'],
                category_ids=batch['category_ids'],
                rgb=batch.get('rgb'),
                depth=batch.get('depth'),
                timesteps=batch.get('timestep'),
                return_loss=True
            )

            # Compute loss (output contains the loss)
            if output.noise_pred is not None:
                loss = output.noise_pred
            else:
                # If no explicit loss returned, assume it's embedded
                loss = torch.tensor(0.0, device=self.device)

            # For synthetic data, we get loss from diffusion.training_losses
            # In practice, we need to modify the forward to return loss
            # Here we compute it directly
            with torch.no_grad():
                # Simple placeholder - in real code, model returns loss
                target = batch['target_map']
                B, C, T, H, W = target.shape
                noise = torch.randn_like(target)
                t = torch.randint(0, 1000, (B,), device=self.device)
                loss = self.model.diffusion.training_losses(
                    self.model.unet,
                    target,
                    t,
                    output.observation_features,
                    noise
                )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'avg': total_loss / num_batches})

            # Log
            if self.global_step % self.log_frequency == 0:
                self._log({'step_loss': loss.item(), 'global_step': self.global_step})

        return total_loss / num_batches

    def validate(self, loader: DataLoader, epoch: int) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Forward
                output = self.model(
                    bev_map=batch['bev_map'],
                    semantic_map=batch['semantic_map'],
                    category_ids=batch['category_ids'],
                    rgb=batch.get('rgb'),
                    depth=batch.get('depth'),
                    timesteps=batch.get('timestep'),
                    return_loss=False
                )

                # Compute loss
                target = batch['target_map']
                B, C, T, H, W = target.shape
                noise = torch.randn_like(target)
                t = torch.randint(0, 1000, (B,), device=self.device)
                loss = self.model.diffusion.training_losses(
                    self.model.unet,
                    target,
                    t,
                    output.observation_features,
                    noise
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        print(f"Checkpoint loaded from {path}")

    def _log(self, metrics: Dict[str, float]):
        """Log metrics."""
        if self.use_wandb:
            wandb.log(metrics)
        print(f"Step {self.global_step}: {metrics}")


class Stage1Trainer(Trainer):
    """
    Stage 1: Pretrain map generation.
    Learns to generate accurate BEV maps from observations.
    """

    def __init__(self, model, config, device="cuda"):
        super().__init__(model, config, device)

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Stage 1 training - focus on map generation quality."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Stage 1 - Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()

            # Stage 1: Learn map generation
            target = batch['target_map']
            B = target.shape[0]

            # Encode conditions
            map_repr = self.model.map_encoder(
                batch['bev_map'],
                batch['semantic_map'],
                batch['category_ids']
            )
            obs_features = self.model.obs_encoder(
                rgb=batch.get('rgb'),
                depth=batch.get('depth')
            )
            if isinstance(obs_features, dict):
                obs_features = obs_features.get('fused', obs_features.get('rgb'))

            condition = self.model.condition_proj(
                torch.cat([map_repr.combined_features, obs_features], dim=-1)
            )

            # Diffusion loss
            noise = torch.randn_like(target)
            t = torch.randint(0, 1000, (B,), device=self.device)
            loss = self.model.diffusion.training_losses(
                self.model.unet,
                target,
                t,
                condition,
                noise,
                loss_type="mse"
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/num_batches:.4f}'})

        return total_loss / num_batches


class Stage2Trainer(Trainer):
    """
    Stage 2: Task-oriented fine-tuning.
    Optimizes map generation for VLN success metrics.
    """

    def __init__(self, model, config, device="cuda"):
        super().__init__(model, config, device)

        # Additional reward head
        self.reward_weight = config.get('reward_weight', 0.1)

    def compute_navigation_reward(
        self,
        generated_map: torch.Tensor,
        target_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute navigation-related reward.

        Args:
            generated_map: (B, C, T, H, W) generated trajectory
            target_map: (B, C, T, H, W) target trajectory

        Returns:
            reward: (B,) per-sample reward
        """
        # Simplified reward: MSE between maps
        mse = torch.mean((generated_map - target_map) ** 2, dim=list(range(1, generated_map.dim())))
        reward = 1.0 / (1.0 + mse)

        if reward.dim() > 1:
            reward = reward.mean(dim=-1)

        return reward

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Stage 2 training - include navigation reward."""
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Stage 2 - Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()

            # Stage 1: Map generation loss
            target = batch['target_map']
            B = target.shape[0]

            map_repr = self.model.map_encoder(
                batch['bev_map'],
                batch['semantic_map'],
                batch['category_ids']
            )
            obs_features = self.model.obs_encoder(
                rgb=batch.get('rgb'),
                depth=batch.get('depth')
            )
            if isinstance(obs_features, dict):
                obs_features = obs_features.get('fused', obs_features.get('rgb'))

            condition = self.model.condition_proj(
                torch.cat([map_repr.combined_features, obs_features], dim=-1)
            )

            noise = torch.randn_like(target)
            t = torch.randint(0, 1000, (B,), device=self.device)
            diffusion_loss = self.model.diffusion.training_losses(
                self.model.unet,
                target,
                t,
                condition,
                noise,
                loss_type="mse"
            )

            # Stage 2: Navigation reward
            with torch.no_grad():
                # Generate maps for reward computation
                sampler = DDIMSampler(
                    self.model.diffusion,
                    self.model.unet,
                    num_timesteps=20
                )
                generated = sampler.sample(
                    condition=condition,
                    shape=(self.model.map_channels, 1, self.model.map_size, self.model.map_size),
                    device=self.device
                )

                reward = self.compute_navigation_reward(generated, target)
                reward_loss = -reward.mean() * self.reward_weight

            # Combined loss
            loss = diffusion_loss + reward_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += diffusion_loss.item()
            total_reward += reward.mean().item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'diff_loss': f'{diffusion_loss.item():.4f}',
                'reward': f'{reward.mean().item():.4f}'
            })

        return total_loss / num_batches


from models.diffusion.ddim_sampler import DDIMSampler


def train_mapdream(
    model: Dual-MapNavDiffusion,
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    use_wandb: bool = False
):
    """
    Convenience function to train Dual-MapNav model.

    Args:
        model: Dual-MapNavDiffusion model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device to train on
        use_wandb: Whether to use wandb logging
    """
    # Create dataset
    dataset = SyntheticMapDataset(
        num_samples=1000,
        map_size=64,
        num_classes=10
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Create trainer
    config = {
        'learning_rate': learning_rate,
        'max_epochs': num_epochs,
        'log_frequency': 100,
        'save_frequency': 10,
        'checkpoint_dir': './checkpoints',
        'use_wandb': use_wandb
    }

    trainer = Trainer(model, config, device=device)
    trainer.train(train_loader)


def resume_training(
    model: Dual-MapNavDiffusion,
    checkpoint_path: str,
    device: str = "cuda"
):
    """
    Resume training from a checkpoint.

    Args:
        model: Dual-MapNavDiffusion model
        checkpoint_path: Path to checkpoint
        device: Device to use
    """
    trainer = Trainer(model, {}, device=device)
    trainer.load_checkpoint(checkpoint_path)

    dataset = SyntheticMapDataset(num_samples=1000)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    trainer.train(train_loader)
