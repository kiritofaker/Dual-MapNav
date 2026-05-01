"""Synthetic data generator for Dual-MapNav."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset


class SyntheticMapDataset(Dataset):
    """
    Synthetic dataset for Dual-MapNav training.

    Generates random BEV maps, semantic maps, and observations
    for training the map prediction model.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        map_size: int = 64,
        num_classes: int = 10,
        bev_channels: int = 3,
        image_size: int = 224,
        num_categories: int = 10,
        split: str = "train"
    ):
        self.num_samples = num_samples
        self.map_size = map_size
        self.num_classes = num_classes
        self.bev_channels = bev_channels
        self.image_size = image_size
        self.num_categories = num_categories
        self.split = split

        # Categories for text
        self.category_names = [
            "chair", "table", "plant", "bed", "sofa",
            "cabinet", "sink", "toilet", "bathtub", "refrigerator"
        ]

        np.random.seed(42 if split == "train" else 123)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a single synthetic sample.

        Returns:
            Dictionary with:
            - bev_map: (3, H, W) bird's-eye view
            - semantic_map: (num_classes, H, W) semantic segmentation
            - rgb: (3, H, W) simulated RGB image
            - depth: (1, H, W) simulated depth image
            - category_ids: (N,) category indices
            - instruction: (L,) tokenized instruction
            - target_map: (C, 1, H, W) target map for prediction
        """
        sample = {}

        # Generate BEV map (traversability + obstacles)
        bev_map = self._generate_bev_map()
        sample['bev_map'] = bev_map

        # Generate semantic map
        semantic_map = self._generate_semantic_map(bev_map)
        sample['semantic_map'] = semantic_map

        # Simulate RGB observation (random but correlated with BEV)
        rgb = self._generate_rgb_from_bev(bev_map)
        sample['rgb'] = rgb

        # Simulate depth (random for now)
        depth = torch.randn(1, self.image_size, self.image_size) * 0.5 + 3.0
        depth = depth.clamp(0.1, 10.0)
        sample['depth'] = depth

        # Category IDs (random subset of categories present in scene)
        num_present = np.random.randint(1, min(5, self.num_categories))
        present_cats = np.random.choice(self.num_categories, num_present, replace=False)
        category_ids = torch.zeros(self.num_categories)
        category_ids[present_cats] = 1.0
        sample['category_ids'] = category_ids.long()

        # Simulated instruction tokens
        instruction = self._generate_instruction(present_cats)
        sample['instruction'] = instruction

        # Target map for diffusion training (next-step BEV prediction)
        # For synthetic data, predict slightly modified BEV
        target_map = self._generate_target_map(bev_map)
        sample['target_map'] = target_map

        # Timestep for diffusion
        if self.split == "train":
            timestep = torch.randint(0, 1000, (1,)).item()
            sample['timestep'] = timestep
        else:
            sample['timestep'] = 0

        return sample

    def _generate_bev_map(self) -> torch.Tensor:
        """Generate a synthetic BEV map."""
        # Simple approach: random blobs for traversable areas
        bev = torch.zeros(3, self.map_size, self.map_size)

        # Channel 0: Traversability (1 = free, 0 = blocked)
        # Add random walk pattern for realistic paths
        traversable = self._generate_traversable_map()
        bev[0] = traversable

        # Channel 1: Distance to nearest obstacle
        bev[1] = self._generate_distance_map(traversable)

        # Channel 2: Gradient/orientation
        bev[2] = self._generate_gradient_map(traversable)

        return bev

    def _generate_traversable_map(self) -> torch.Tensor:
        """Generate a traversable area map using random walks."""
        grid = torch.zeros(self.map_size, self.map_size)

        # Start with some blocked regions (obstacles)
        num_obstacles = np.random.randint(3, 10)
        for _ in range(num_obstacles):
            cx = np.random.randint(0, self.map_size)
            cy = np.random.randint(0, self.map_size)
            radius = np.random.randint(3, 10)
            for i in range(self.map_size):
                for j in range(self.map_size):
                    if (i - cx) ** 2 + (j - cy) ** 2 < radius ** 2:
                        grid[i, j] = 1

        # Add some rectangular obstacles
        num_rects = np.random.randint(2, 5)
        for _ in range(num_rects):
            x1 = np.random.randint(0, self.map_size - 5)
            y1 = np.random.randint(0, self.map_size - 5)
            w = np.random.randint(3, 10)
            h = np.random.randint(3, 10)
            grid[x1:x1+w, y1:y1+h] = 1

        return 1 - grid  # 1 = traversable, 0 = blocked

    def _generate_distance_map(self, traversable: torch.Tensor) -> torch.Tensor:
        """Generate distance transform to obstacles."""
        # Simple approximation using blur
        from scipy.ndimage import distance_transform_edt

        traversable_np = traversable.numpy().astype(np.float32)
        dist = distance_transform_edt(traversable_np)
        dist = torch.from_numpy(dist).float()

        # Normalize
        if dist.max() > 0:
            dist = dist / dist.max()

        return dist

    def _generate_gradient_map(self, traversable: torch.Tensor) -> torch.Tensor:
        """Generate gradient/orientation map."""
        # Sobel-like gradient
        dx = torch.zeros_like(traversable)
        dy = torch.zeros_like(traversable)

        dx[1:-1, :] = traversable[2:, :] - traversable[:-2, :]
        dy[:, 1:-1] = traversable[:, 2:] - traversable[:, :-2]

        gradient = torch.sqrt(dx ** 2 + dy ** 2)
        return gradient

    def _generate_semantic_map(self, bev_map: torch.Tensor) -> torch.Tensor:
        """Generate semantic segmentation map."""
        semantic = torch.zeros(self.num_classes, self.map_size, self.map_size)

        # Randomly place some semantic objects
        num_objects = np.random.randint(2, 6)
        for _ in range(num_objects):
            cat_idx = np.random.randint(0, self.num_classes)
            cx = np.random.randint(5, self.map_size - 5)
            cy = np.random.randint(5, self.map_size - 5)
            w = np.random.randint(3, 8)
            h = np.random.randint(3, 8)

            x1, x2 = max(0, cx - w//2), min(self.map_size, cx + w//2)
            y1, y2 = max(0, cy - h//2), min(self.map_size, cy + h//2)

            # Only place where traversable
            if bev_map[0, x1:x2, y1:y2].mean() > 0.5:
                semantic[cat_idx, x1:x2, y1:y2] = 1.0

        return semantic

    def _generate_rgb_from_bev(self, bev_map: torch.Tensor) -> torch.Tensor:
        """Generate simulated RGB from BEV (simplified)."""
        # Just use BEV as a guide and create correlated random patterns
        rgb = torch.randn(3, self.image_size, self.image_size) * 0.3

        # Resize BEV to image size and use as guide
        bev_small = torch.nn.functional.interpolate(
            bev_map.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Corrupt with structure
        rgb = rgb + bev_small[:3] * 0.5

        # Normalize to ImageNet-like range
        rgb = rgb.clamp(-2, 2)

        return rgb

    def _generate_target_map(self, bev_map: torch.Tensor) -> torch.Tensor:
        """
        Generate target map for prediction.
        For synthetic data, this is a slightly modified version of input BEV.
        """
        # Add small perturbation
        target = bev_map.clone()

        # Small random translation
        shift_x = np.random.randint(-3, 4)
        shift_y = np.random.randint(-3, 4)

        target = torch.roll(target, shifts=(shift_x, shift_y), dims=(1, 2))

        # Add small noise
        target = target + torch.randn_like(target) * 0.05

        # Clip to valid range
        target = target.clamp(0, 1)

        # Reshape for diffusion (add frame dim)
        target = target.unsqueeze(2)  # (C, 1, H, W)

        return target

    def _generate_instruction(self, present_cats: np.ndarray) -> torch.Tensor:
        """Generate simulated instruction tokens."""
        templates = [
            "Go to the {}",
            "Navigate to the {}",
            "Find the {}",
            "Move towards the {}",
            "Head to the {}",
        ]

        words = [self.category_names[i] for i in present_cats[:3]]
        template = templates[np.random.randint(0, len(templates))]
        text = template.format(words[0] if words else "target")

        # Simple tokenization (in practice, use proper tokenizer)
        tokens = torch.randint(0, 1000, (10,))
        tokens[0] = 2  # BOS
        tokens[-1] = 3  # EOS

        return tokens


def generate_synthetic_batch(
    batch_size: int = 8,
    map_size: int = 64,
    num_classes: int = 10,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Generate a batch of synthetic data for testing.

    Args:
        batch_size: Number of samples
        map_size: Size of map
        num_classes: Number of semantic classes
        device: Device to place tensors on

    Returns:
        Dictionary of synthetic tensors
    """
    dataset = SyntheticMapDataset(
        num_samples=batch_size,
        map_size=map_size,
        num_classes=num_classes
    )

    batch = {}
    for i in range(batch_size):
        sample = dataset[i]
        for key, val in sample.items():
            if key not in batch:
                batch[key] = []
            batch[key].append(val)

    # Stack tensors
    for key in batch:
        batch[key] = torch.stack(batch[key])

    return batch


class SyntheticTrajectoryDataset(Dataset):
    """
    Dataset for map trajectory prediction.
    Generates sequences of maps for video diffusion training.
    """

    def __init__(
        self,
        num_samples: int = 500,
        map_size: int = 64,
        num_frames: int = 16,
        num_classes: int = 10
    ):
        self.num_samples = num_samples
        self.map_size = map_size
        self.num_frames = num_frames
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a trajectory of maps.

        Returns:
            Dictionary with trajectory data
        """
        # Generate a base map
        base_bev = torch.rand(3, self.map_size, self.map_size)

        # Generate trajectory as slight variations
        trajectory = []
        for t in range(self.num_frames):
            # Each frame is slightly different (simulating robot movement)
            shift_x = int(t * 0.2)
            shift_y = int(t * 0.1)
            frame = torch.roll(base_bev, shifts=(shift_x, shift_y), dims=(1, 2))
            trajectory.append(frame)

        trajectory = torch.stack(trajectory, dim=1)  # (C, T, H, W)

        return {
            'trajectory': trajectory,
            'timestep': torch.randint(0, 1000, (1,)).item()
        }
