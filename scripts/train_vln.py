"""Training script for Dual-MapNav with VLNTube/InteriorNav data."""

import torch
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.map_predictor import Dual-MapNavDiffusion
from data.vln_dataset import InteriorNavDataset
from training.vln_trainer import VLNTrainer, VLNStage1Trainer, VLNStage2Trainer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dual-MapNav with VLN data")

    # Model
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--map-channels", type=int, default=16)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)  # VLN data is larger
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])

    # Data
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to VLNTube/InteriorNav format data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--instruction-type", type=str, default="all",
                        choices=["fine", "coarse", "all"],
                        help="Type of instructions to use")
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
    print(f"Stage {args.stage} training with VLN data")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}, Instruction type: {args.instruction_type}")

    # Check if data exists
    if not os.path.exists(args.data_root):
        print(f"Error: Data root does not exist: {args.data_root}")
        print("Please provide a valid path to VLNTube/InteriorNav format data.")
        print("You can download pre-built data from:")
        print("  https://huggingface.co/datasets/Eyz/CaffeEclipse")
        return

    # Create model
    model = Dual-MapNavDiffusion(
        map_size=args.map_size,
        num_map_frames=args.num_frames,
        diffusion_channels=args.map_channels
    )

    # Create dataset
    dataset = InteriorNavDataset(
        data_root=args.data_root,
        split=args.split,
        instruction_type=args.instruction_type,
        image_size=224,
        map_size=args.map_size,
        cache_in_memory=False
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
        'use_wandb': args.wandb,
        'map_size': args.map_size,
        'resolution': 0.25,
        'map_range': 16.0
    }

    if args.stage == 1:
        trainer = VLNStage1Trainer(model, config, device=args.device)
    else:
        trainer = VLNStage2Trainer(model, config, device=args.device)

    # Train
    trainer.train(train_loader)

    print("Training complete!")


if __name__ == "__main__":
    main()
