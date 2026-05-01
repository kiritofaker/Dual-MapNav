"""Training pipeline for Dual-MapNav with VLN data support."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Union
import os
from tqdm import tqdm

from models.map_predictor import Dual-MapNavDiffusion
from data.synthetic_data import SyntheticMapDataset
from data.vln_dataset import InteriorNavDataset, BEVMapGenerator


class VLNTrainer:
    """
    Trainer for Dual-MapNav with VLNTube/InteriorNav data support.
    Handles both synthetic and real VLN data.
    """

    def __init__(
        self,
        model: Dual-MapNavDiffusion,
        config: Dict[str, Any],
        device: str = "cuda",
        use_bev_generator: bool = True
    ):
        self.model = model
        self.config = config
        self.device = device
        self.use_bev_generator = use_bev_generator

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

        # BEV map generator for VLN data
        if use_bev_generator:
            self.bev_generator = BEVMapGenerator(
                map_size=config.get('map_size', 64),
                resolution=config.get('resolution', 0.25),
                map_range=config.get('map_range', 16.0)
            )
        else:
            self.bev_generator = None

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
            batch = self._prepare_batch(batch)

            # Forward pass
            self.optimizer.zero_grad()

            # Determine data type and process accordingly
            if 'bev_map' in batch and 'rgb' not in batch:
                # Synthetic data
                loss = self._train_step_synthetic(batch)
            else:
                # VLN data
                loss = self._train_step_vln(batch)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss / num_batches:.4f}'})

            # Log
            if self.global_step % self.log_frequency == 0:
                self._log({'step_loss': loss.item(), 'global_step': self.global_step})

        return total_loss / num_batches

    def _prepare_batch(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        prepared = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.device)
            elif isinstance(v, dict):
                prepared[k] = {kk: vv.to(self.device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            else:
                prepared[k] = v
        return prepared

    def _train_step_synthetic(self, batch: Dict) -> torch.Tensor:
        """Training step for synthetic data."""
        output = self.model(
            bev_map=batch['bev_map'],
            semantic_map=batch['semantic_map'],
            category_ids=batch['category_ids'],
            rgb=batch.get('rgb'),
            depth=batch.get('depth'),
            timesteps=batch.get('timestep'),
            return_loss=True
        )

        # Get target and compute loss
        target = batch['target_map']
        B, C, T, H, W = target.shape

        noise = torch.randn_like(target)
        t = torch.randint(0, 1000, (B,), device=self.device)

        loss = self.model.diffusion.training_losses(
            self.model.unet,
            target,
            t,
            output.observation_features,
            noise,
            loss_type="mse"
        )

        return loss

    def _train_step_vln(self, batch: Dict) -> torch.Tensor:
        """
        Training step for VLN data.

        VLN batch contains:
        - rgb: (B, T, 3, H, W) RGB frames
        - depth: (B, T, 1, H, W) depth frames
        - positions: (B, T, 3) positions
        - yaws: (B, T) yaw angles
        - actions: (B, T) actions
        - progress: (B, T) progress values
        - instruction_text: list of strings
        - goal_position: (B, 3) goal positions
        """
        rgb = batch.get('rgb')  # (B, T, 3, H, W)
        depth = batch.get('depth')  # (B, T, 1, H, W)
        positions = batch.get('positions')  # (B, T, 3)
        goal_position = batch.get('goal_position')  # (B, 3)

        if rgb is None or positions is None:
            # Fallback to synthetic-like training
            return self._train_step_synthetic(batch)

        B, T = rgb.shape[:2]

        # Process sequence - use first frame as observation
        # For map prediction, we want to predict future maps
        obs_rgb = rgb[:, 0]  # (B, 3, H, W)
        obs_depth = depth[:, 0] if depth is not None else None  # (B, 1, H, W)

        # Generate BEV map from trajectory
        if self.bev_generator is not None:
            # Use first position as start
            start_pos = positions[:, 0]  # (B, 3)
            bev_maps = []
            for i in range(B):
                bev = self.bev_generator.generate_bev_from_trajectory(
                    positions[i],
                    goal_position[i]
                )
                bev_maps.append(bev)
            bev_map = torch.stack(bev_maps).to(self.device)  # (B, 3, map_size, map_size)
        else:
            # Fallback: create dummy BEV
            bev_map = torch.zeros(B, 3, 64, 64, device=self.device)

        # Semantic map - placeholder (in real VLN data, this would come from scene graph)
        semantic_map = torch.zeros(B, 10, 64, 64, device=self.device)

        # Category IDs - placeholder
        category_ids = torch.zeros(B, 10, dtype=torch.long, device=self.device)

        # For VLN, target is the BEV map at the end of trajectory or goal
        # Simplified: predict the full trajectory BEV
        target_maps = []
        for i in range(B):
            target_bev = self.bev_generator.generate_bev_from_trajectory(
                positions[i],
                goal_position[i]
            )
            target_maps.append(target_bev)
        target_map = torch.stack(target_maps).unsqueeze(2).to(self.device)  # (B, C, 1, H, W)

        # Encode observation
        map_repr = self.model.map_encoder(bev_map, semantic_map, category_ids)

        # Use average RGB features across sequence as observation
        obs_features_list = []
        for t_idx in range(min(T, 5)):  # Use up to 5 frames
            frame_rgb = rgb[:, t_idx]
            obs_feat = self.model.obs_encoder(rgb=frame_rgb)
            if isinstance(obs_feat, dict):
                obs_feat = obs_feat.get('fused', obs_feat.get('rgb'))
            obs_features_list.append(obs_feat)

        # Average observation features
        obs_features = torch.stack(obs_features_list).mean(dim=0)

        # Combine conditions
        condition = self.model.condition_proj(
            torch.cat([map_repr.combined_features, obs_features], dim=-1)
        )

        # Compute diffusion loss
        noise = torch.randn_like(target_map)
        t = torch.randint(0, 1000, (B,), device=self.device)

        loss = self.model.diffusion.training_losses(
            self.model.unet,
            target_map,
            t,
            condition,
            noise,
            loss_type="mse"
        )

        return loss

    def validate(self, loader: DataLoader, epoch: int) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                batch = self._prepare_batch(batch)

                if 'bev_map' in batch and 'rgb' not in batch:
                    loss = self._train_step_synthetic(batch)
                else:
                    loss = self._train_step_vln(batch)

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


class VLNStage1Trainer(VLNTrainer):
    """Stage 1: Pretrain map generation with VLN data."""

    def _train_step_vln(self, batch: Dict) -> torch.Tensor:
        """Stage 1 focuses on map generation quality."""
        # Same as base VLN training
        return super()._train_step_vln(batch)


class VLNStage2Trainer(VLNTrainer):
    """Stage 2: Task-oriented fine-tuning with VLN rewards."""

    def __init__(self, model, config, device="cuda", use_bev_generator=True):
        super().__init__(model, config, device, use_bev_generator)
        self.reward_weight = config.get('reward_weight', 0.1)

    def compute_navigation_reward(
        self,
        generated_map: torch.Tensor,
        target_map: torch.Tensor,
        predicted_goal: torch.Tensor,
        true_goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute navigation-related reward.

        Args:
            generated_map: (B, C, 1, H, W) predicted BEV
            target_map: (B, C, 1, H, W) target BEV
            predicted_goal: (B, 3) predicted goal position
            true_goal: (B, 3) true goal position

        Returns:
            reward: (B,) per-sample reward
        """
        # Map quality reward
        map_mse = torch.mean((generated_map - target_map) ** 2, dim=list(range(1, generated_map.dim())))
        map_reward = 1.0 / (1.0 + map_mse)

        # Goal distance reward
        goal_dist = torch.norm(predicted_goal - true_goal, dim=-1)
        goal_reward = 1.0 / (1.0 + goal_dist)

        # Combined reward
        reward = map_reward * 0.5 + goal_reward * 0.5

        if reward.dim() > 1:
            reward = reward.mean(dim=-1)

        return reward

    def _train_step_vln(self, batch: Dict) -> torch.Tensor:
        """Stage 2 includes navigation reward."""
        rgb = batch.get('rgb')
        depth = batch.get('depth')
        positions = batch.get('positions')
        goal_position = batch.get('goal_position')

        if rgb is None or positions is None:
            return super()._train_step_vln(batch)

        B, T = rgb.shape[:2]

        # Generate observation features
        obs_rgb = rgb[:, 0]
        obs_depth = depth[:, 0] if depth is not None else None

        # Generate BEV maps
        bev_maps = []
        target_maps = []
        for i in range(B):
            bev = self.bev_generator.generate_bev_from_trajectory(
                positions[i], goal_position[i]
            )
            bev_maps.append(bev)
            target_maps.append(bev)

        bev_map = torch.stack(bev_maps).to(self.device)
        target_map = torch.stack(target_maps).unsqueeze(2).to(self.device)

        # Encode
        semantic_map = torch.zeros(B, 10, 64, 64, device=self.device)
        category_ids = torch.zeros(B, 10, dtype=torch.long, device=self.device)

        map_repr = self.model.map_encoder(bev_map, semantic_map, category_ids)

        # Observation features
        obs_feat = self.model.obs_encoder(rgb=obs_rgb)
        if isinstance(obs_feat, dict):
            obs_feat = obs_feat.get('fused', obs_feat.get('rgb'))

        condition = self.model.condition_proj(
            torch.cat([map_repr.combined_features, obs_feat], dim=-1)
        )

        # Diffusion loss
        noise = torch.randn_like(target_map)
        t = torch.randint(0, 1000, (B,), device=self.device)
        diffusion_loss = self.model.diffusion.training_losses(
            self.model.unet, target_map, t, condition, noise, loss_type="mse"
        )

        # Reward computation (simplified)
        with torch.no_grad():
            from models.diffusion.ddim_sampler import DDIMSampler
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

            # Simple reward: compare generated to target
            reward = torch.mean(torch.abs(generated - target_map))
            reward_loss = -reward.mean() * self.reward_weight

        # Combined loss
        loss = diffusion_loss + reward_loss

        return loss


def create_vln_trainer(
    model: Dual-MapNavDiffusion,
    config: Dict[str, Any],
    device: str = "cuda",
    stage: int = 1
) -> VLNTrainer:
    """Factory function to create VLN trainer."""
    if stage == 1:
        return VLNStage1Trainer(model, config, device)
    else:
        return VLNStage2Trainer(model, config, device)


def train_with_vln_data(
    model: Dual-MapNavDiffusion,
    data_root: str,
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    split: str = "train",
    instruction_type: str = "all",
    use_wandb: bool = False
):
    """
    Convenience function to train Dual-MapNav with VLN data.

    Args:
        model: Dual-MapNavDiffusion model
        data_root: Path to VLNTube/InteriorNav format data
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device to train on
        split: Data split ("train", "val", "test")
        instruction_type: "fine", "coarse", or "all"
        use_wandb: Whether to use wandb logging
    """
    # Create dataset
    dataset = InteriorNavDataset(
        data_root=data_root,
        split=split,
        instruction_type=instruction_type,
        image_size=224,
        map_size=64
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
        'use_wandb': use_wandb,
        'map_size': 64
    }

    trainer = VLNTrainer(model, config, device=device)
    trainer.train(train_loader)

    return trainer
