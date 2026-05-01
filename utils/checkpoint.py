"""Checkpoint utilities."""

import torch
import os
from typing import Dict, Any, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    path: str
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        step: Global step
        metrics: Dictionary of metrics
        config: Configuration dict
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'global_step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Checkpoint path
        model: PyTorch model
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load on

    Returns:
        Dictionary with checkpoint info
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Step: {checkpoint.get('global_step', 'N/A')}")
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")

    return checkpoint


class CheckpointManager:
    """
    Manages model checkpoints with automatic saving.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str = "./checkpoints",
        max_to_keep: int = 5
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.checkpoints = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        name: str,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None
    ):
        """Save a checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        save_checkpoint(
            self.model,
            self.optimizer,
            None,
            epoch,
            step,
            metrics or {},
            config or {},
            path
        )

        self.checkpoints.append(path)
        self._cleanup_old()

    def _cleanup_old(self):
        """Remove old checkpoints keeping only max_to_keep."""
        if len(self.checkpoints) > self.max_to_keep:
            to_remove = self.checkpoints[:-self.max_to_keep]
            for path in to_remove:
                if os.path.exists(path):
                    os.remove(path)
            self.checkpoints = self.checkpoints[-self.max_to_keep:]

    def load_best(self, name: str = "best"):
        """Load the best checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        if os.path.exists(path):
            return load_checkpoint(path, self.model, self.optimizer)
        else:
            print(f"No best checkpoint found at {path}")
            return None
