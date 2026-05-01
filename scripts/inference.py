"""Inference script for Dual-MapNav."""

import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.map_predictor import Dual-MapNavDiffusion
from inference.map_inference import MapInference
from data.synthetic_data import generate_synthetic_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Run Dual-MapNav inference")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--num-steps", type=int, default=50, help="DDIM sampling steps")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Running Dual-MapNav inference on {args.device}")

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
    else:
        print("No checkpoint provided, using random weights")

    # Create inference
    inference = MapInference(
        model=model,
        device=args.device,
        num_sampling_steps=args.num_steps
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate samples
    for i in range(args.num_samples):
        print(f"\nGenerating sample {i+1}/{args.num_samples}")

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

        print(f"Generated map shape: {generated.shape}")
        print(f"BEV map shape: {batch['bev_map'].shape}")
        print(f"Target map shape: {batch['target_map'].shape}")

        # Save outputs
        torch.save({
            'generated': generated,
            'bev_map': batch['bev_map'],
            'target_map': batch['target_map'],
            'semantic_map': batch['semantic_map']
        }, os.path.join(args.output_dir, f"sample_{i}.pt"))

        print(f"Saved sample {i+1} to {args.output_dir}/sample_{i}.pt")

    print(f"\nDone! Generated {args.num_samples} samples.")


if __name__ == "__main__":
    main()
