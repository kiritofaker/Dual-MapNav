"""Evaluation script for Dual-MapNav."""

import torch
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.map_predictor import Dual-MapNavDiffusion
from inference.map_inference import MapInference, visualize_map
from data.synthetic_data import generate_synthetic_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dual-MapNav")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./outputs")

    return parser.parse_args()


def compute_metrics(generated, target):
    """Compute evaluation metrics."""
    mse = torch.mean((generated - target) ** 2).item()
    mae = torch.mean(torch.abs(generated - target)).item()

    return {
        'mse': mse,
        'mae': mae
    }


def main():
    args = parse_args()

    print(f"Evaluating Dual-MapNav on {args.device}")

    # Create model
    model = Dual-MapNavDiffusion(
        map_size=64,
        num_map_frames=16,
        diffusion_channels=16
    )

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Create inference
    inference = MapInference(
        model=model,
        device=args.device,
        num_sampling_steps=50
    )

    # Generate samples
    all_metrics = []

    for i in range(args.num_samples):
        # Generate synthetic batch
        batch = generate_synthetic_batch(
            batch_size=1,
            map_size=64,
            device=args.device
        )

        # Generate map
        generated = inference.generate_map(
            bev_map=batch['bev_map'],
            semantic_map=batch['semantic_map'],
            category_ids=batch['category_ids'],
            rgb=batch.get('rgb'),
            depth=batch.get('depth'),
            use_cfg=True,
            seed=i
        )

        # Get target
        target = batch['target_map']

        # Compute metrics
        metrics = compute_metrics(generated, target)
        all_metrics.append(metrics)

        print(f"Sample {i+1}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

        # Visualize first sample
        if i == 0:
            os.makedirs(args.output_dir, exist_ok=True)

            # Save visualization
            bev_np = batch['bev_map'][0].cpu().numpy()
            gen_np = generated[0, :3].cpu().numpy() if generated.shape[1] >= 3 else generated[0].cpu().numpy()

            print(f"BEV shape: {bev_np.shape}")
            print(f"Generated shape: {gen_np.shape}")

    # Print averaged metrics
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])

    print(f"\nAveraged Metrics:")
    print(f"  MSE: {avg_mse:.4f}")
    print(f"  MAE: {avg_mae:.4f}")


if __name__ == "__main__":
    main()
