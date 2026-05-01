"""Training script for Dual-MapNav."""

import torch
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.map_predictor import Dual-MapNavDiffusion
from data.synthetic_data import SyntheticMapDataset
from training.trainer import Trainer, Stage1Trainer, Stage2Trainer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dual-MapNav")

    # Model
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--map-channels", type=int, default=16)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])

    # Data
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training Dual-MapNav on {args.device}")
    print(f"Stage {args.stage} training for {args.epochs} epochs")

    # Create model
    model = Dual-MapNavDiffusion(
        map_size=args.map_size,
        num_map_frames=args.num_frames,
        diffusion_channels=args.map_channels
    )

    # Create dataset
    dataset = SyntheticMapDataset(
        num_samples=args.num_samples,
        map_size=args.map_size
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Create trainer
    config = {
        'learning_rate': args.lr,
        'max_epochs': args.epochs,
        'log_frequency': 100,
        'save_frequency': 10,
        'checkpoint_dir': args.checkpoint_dir,
        'use_wandb': args.wandb
    }

    if args.stage == 1:
        trainer = Stage1Trainer(model, config, device=args.device)
    else:
        trainer = Stage2Trainer(model, config, device=args.device)

    # Train
    trainer.train(train_loader)

    print("Training complete!")


if __name__ == "__main__":
    main()
