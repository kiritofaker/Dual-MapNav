"""Evaluation script for Dual-MapNav with VLNTube data."""

import torch
import argparse
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.map_predictor import Dual-MapNavDiffusion
from data.vln_dataset import InteriorNavDataset, BEVMapGenerator
from inference.map_inference import MapInference
from utils.metrics import compute_vln_metrics, compute_map_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dual-MapNav with VLN data")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--map-size", type=int, default=64)

    # Data
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=100)

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./outputs")

    return parser.parse_args()


def evaluate_single_trajectory(
    inference: MapInference,
    episode: dict,
    bev_generator: BEVMapGenerator,
    device: str
) -> dict:
    """Evaluate a single VLN trajectory."""
    traj_dir = episode.get('traj_dir')
    if not traj_dir:
        return None

    traj_path = Path(traj_dir)

    # Load parquet for ground truth
    import pandas as pd
    parquet_path = traj_path / "data" / "chunk-000" / "episode_000000.parquet"
    if not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path)
    positions = np.array(df['observation.robot_position'].tolist())
    goal_pos = positions[-1] if len(positions) > 0 else np.zeros(3)

    # Load RGB frames
    rgb_path = traj_path / "videos" / "chunk-000" / "observation.images.rgb" / "rgb.npy"
    if not rgb_path.exists():
        return None

    rgb_frames = np.load(rgb_path)
    if len(rgb_frames) > 50:
        indices = np.linspace(0, len(rgb_frames) - 1, 50, dtype=int)
        rgb_frames = rgb_frames[indices]

    # Load instruction
    meta_path = traj_path / "meta" / "episodes.jsonl"
    instruction = ""
    if meta_path.exists():
        import json
        with open(meta_path, 'r') as f:
            meta = json.loads(f.readline())
            instruction = meta.get('instruction_text', '')

    # Generate BEV map
    positions_tensor = torch.from_numpy(positions).float()
    goal_tensor = torch.from_numpy(goal_pos).float()
    bev_map = bev_generator.generate_bev_from_trajectory(positions_tensor, goal_tensor)
    bev_map = bev_map.unsqueeze(0).to(device)

    # Use first RGB frame
    first_rgb = torch.from_numpy(rgb_frames[0]).permute(2, 0, 1).float() / 255.0
    first_rgb = torch.nn.functional.interpolate(
        first_rgb.unsqueeze(0),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    # Generate prediction
    semantic_map = torch.zeros(1, 10, 64, 64, device=device)
    category_ids = torch.zeros(1, 10, dtype=torch.long, device=device)

    generated = inference.generate_map(
        bev_map=bev_map,
        semantic_map=semantic_map,
        category_ids=category_ids,
        rgb=first_rgb.unsqueeze(0),
        use_cfg=True
    )

    # Compute metrics
    target_map = bev_map.unsqueeze(2)
    map_metrics = compute_map_metrics(generated, target_map)

    return {
        'map_mse': map_metrics.get('mse', 0),
        'map_mae': map_metrics.get('mae', 0),
        'instruction': instruction,
        'num_steps': len(rgb_frames)
    }


def main():
    args = parse_args()

    print(f"Evaluating Dual-MapNav on {args.device}")
    print(f"Data root: {args.data_root}")

    if not os.path.exists(args.data_root):
        print(f"Error: Data root does not exist: {args.data_root}")
        return

    # Create model
    model = Dual-MapNavDiffusion(
        map_size=args.map_size,
        num_map_frames=16,
        diffusion_channels=16
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")

    inference = MapInference(model=model, device=args.device, num_sampling_steps=50)
    bev_generator = BEVMapGenerator(map_size=args.map_size, resolution=0.25, map_range=16.0)

    dataset = InteriorNavDataset(
        data_root=args.data_root,
        split=args.split,
        instruction_type="all",
        image_size=224,
        map_size=args.map_size
    )

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []
    num_evaluated = 0

    print(f"Evaluating up to {args.num_samples} episodes...")

    for i in range(min(args.num_samples, len(dataset))):
        episode = dataset[i]
        episode_id = episode.get('episode_id', f'episode_{i}')
        print(f"\nEvaluating {i+1}/{min(args.num_samples, len(dataset))}: {episode_id}")

        metrics = evaluate_single_trajectory(inference, episode, bev_generator, args.device)

        if metrics:
            all_metrics.append(metrics)
            num_evaluated += 1
            print(f"  Map MSE: {metrics['map_mse']:.4f}, MAE: {metrics['map_mae']:.4f}")

    if all_metrics:
        avg_mse = np.mean([m['map_mse'] for m in all_metrics])
        avg_mae = np.mean([m['map_mae'] for m in all_metrics])
        print(f"\n{'='*60}")
        print(f"Results ({num_evaluated} episodes)")
        print(f"  Map MSE: {avg_mse:.4f}")
        print(f"  Map MAE: {avg_mae:.4f}")


if __name__ == "__main__":
    main()
