"""VLNTube/InteriorNav format dataset loader for Dual-MapNav."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import gzip
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
import clip
from PIL import Image


class InteriorNavDataset(Dataset):
    """
    Dataset loader for VLNTube InteriorNav format.

    Expected directory structure:
    <data_root>/
    ├── <scene_id>/
    │   └── <goal>_<start>/
    │       ├── data/chunk-000/
    │       │   └── episode_000000.parquet  # Positions, orientations, actions
    │       ├── videos/chunk-000/
    │       │   ├── observation.images.rgb/rgb.npy
    │       │   └── observation.images.depth/depth.npy
    │       └── meta/
    │           ├── episodes.jsonl
    │           ├── info.json
    │           └── tasks.jsonl
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        instruction_type: str = "all",  # "fine", "coarse", or "all"
        max_frames: int = 50,
        image_size: int = 224,
        map_size: int = 64,
        transform_rgb: bool = True,
        cache_in_memory: bool = False,
        preload_videos: bool = False
    ):
        """
        Args:
            data_root: Root directory containing scene folders
            split: "train" or "val" or "test"
            instruction_type: "fine", "coarse", or "all"
            max_frames: Maximum number of frames to sample per trajectory
            image_size: Size to resize images to
            map_size: Size for BEV map
            transform_rgb: Whether to normalize RGB images
            cache_in_memory: Whether to cache loaded data in memory
            preload_videos: Whether to preload all videos into memory
        """
        self.data_root = Path(data_root)
        self.split = split
        self.instruction_type = instruction_type
        self.max_frames = max_frames
        self.image_size = image_size
        self.map_size = map_size
        self.transform_rgb = transform_rgb
        self.cache_in_memory = cache_in_memory
        self.preload_videos = preload_videos

        # Cache for loaded data
        self.cache = {} if cache_in_memory else None
        self.video_cache = {} if preload_videos else None

        # Load episode list
        self.episodes = self._load_episodes()

        # CLIP model for vision-language features
        self.clip_model = None
        self.clip_preprocess = None

    def _load_episodes(self) -> List[Dict]:
        """Load all episodes from the dataset."""
        episodes = []

        # Try to find JSON metadata files
        possible_roots = [
            self.data_root.parent / "raw_data" / self.split,
            self.data_root / "raw_data" / self.split,
            self.data_root,
        ]

        json_file = None
        for root in possible_roots:
            if self.instruction_type == "all":
                fine_file = root / f"{self.split}.json"
                coarse_file = root / f"{self.split}.json"
                # Try both fine and coarse
                if fine_file.exists():
                    json_file = fine_file
                    break
            else:
                type_dir = root / f"all_{self.instruction_type}_grained"
                json_file = type_dir / f"{self.split}.json"
                if json_file.exists():
                    break

        if json_file and json_file.exists():
            episodes = self._load_from_json(json_file)
        else:
            # Fallback: scan directory structure
            episodes = self._scan_directory()

        print(f"Loaded {len(episodes)} episodes for {self.split} ({self.instruction_type})")
        return episodes

    def _load_from_json(self, json_file: Path) -> List[Dict]:
        """Load episodes from JSON metadata file."""
        episodes = []

        # Handle .gz files
        if str(json_file).endswith('.gz'):
            with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

        for ep in data.get('episodes', []):
            episode = {
                'scene_id': ep.get('scene_id', ep.get('scan', '')),
                'episode_id': ep.get('episode_id', ''),
                'start_position': ep.get('start_position', [0, 0, 0]),
                'start_rotation': ep.get('start_rotation', [1, 0, 0, 0]),
                'goal_position': ep.get('goals', {}).get('position', [0, 0, 0]),
                'goal_radius': ep.get('goals', {}).get('radius', 3.0),
                'instruction_text': ep.get('instruction', {}).get('instruction_text', ''),
                'instruction_tokens': ep.get('instruction', {}).get('instruction_tokens', []),
                'reference_path': ep.get('reference_path', []),
                'geodesic_distance': ep.get('info', {}).get('geodesic_distance', -1),
                'instruction_type': self.instruction_type
            }
            episodes.append(episode)

        return episodes

    def _scan_directory(self) -> List[Dict]:
        """Scan directory structure to find episodes."""
        episodes = []

        if not self.data_root.exists():
            print(f"Warning: data_root does not exist: {self.data_root}")
            return episodes

        for scene_dir in self.data_root.iterdir():
            if not scene_dir.is_dir():
                continue

            for traj_dir in scene_dir.iterdir():
                if not traj_dir.is_dir():
                    continue

                # Check for required files
                parquet_path = traj_dir / "data" / "chunk-000" / "episode_000000.parquet"
                if not parquet_path.exists():
                    continue

                # Load metadata
                meta_path = traj_dir / "meta" / "episodes.jsonl"
                instruction_text = ""
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.loads(f.readline())
                        instruction_text = meta.get('instruction_text', '')

                # Extract goal and start IDs from directory name
                dir_name = traj_dir.name
                parts = dir_name.split('_')
                if len(parts) >= 2:
                    goal_id = parts[0]
                    start_id = '_'.join(parts[1:])

                    episode = {
                        'scene_id': scene_dir.name,
                        'episode_id': f"{scene_dir.name}_{goal_id}_{start_id}",
                        'traj_dir': str(traj_dir),
                        'instruction_text': instruction_text,
                        'instruction_type': self.instruction_type
                    }
                    episodes.append(episode)

        return episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single episode."""
        episode = self.episodes[idx]

        # Check cache
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        sample = {}

        # Load trajectory data
        traj_dir = episode.get('traj_dir')
        if traj_dir:
            sample = self._load_from_directory(traj_dir, episode)
        else:
            # Use embedded data
            sample = self._load_from_episode(episode)

        # Add instruction
        sample['instruction_text'] = episode.get('instruction_text', '')
        sample['scene_id'] = episode.get('scene_id', '')
        sample['episode_id'] = episode.get('episode_id', '')
        sample['goal_position'] = torch.tensor(episode.get('goal_position', [0, 0, 0]))
        sample['start_position'] = torch.tensor(episode.get('start_position', [0, 0, 0]))

        # Sample frames if needed
        if sample.get('rgb_frames') is not None and len(sample['rgb_frames']) > self.max_frames:
            sample = self._sample_frames(sample)

        # Convert to tensors
        sample = self._prepare_tensors(sample)

        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = sample

        return sample

    def _load_from_directory(self, traj_dir: str, episode: Dict) -> Dict:
        """Load data from directory structure."""
        traj_path = Path(traj_dir)
        sample = {}

        # Load parquet
        parquet_path = traj_path / "data" / "chunk-000" / "episode_000000.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)

            # Extract position and orientation
            positions = np.array(df['observation.robot_position'].tolist())
            orientations = np.array(df['observation.robot_orientation'].tolist())
            yaws = np.array(df['observation.robot_yaw'].tolist())
            actions = np.array(df['observation.action'].tolist())
            progress = np.array(df['observation.progress'].tolist())

            sample['positions'] = positions
            sample['orientations'] = orientations
            sample['yaws'] = yaws
            sample['actions'] = actions
            sample['progress'] = progress
            sample['num_steps'] = len(positions)

        # Load RGB video
        rgb_path = traj_path / "videos" / "chunk-000" / "observation.images.rgb" / "rgb.npy"
        if rgb_path.exists():
            rgb_array = np.load(rgb_path)  # (T, H, W, 3)
            # Resize if needed
            if rgb_array.shape[1] != self.image_size or rgb_array.shape[2] != self.image_size:
                rgb_array = self._resize_images(rgb_array, (self.image_size, self.image_size))
            sample['rgb_frames'] = rgb_array

        # Load Depth video
        depth_path = traj_path / "videos" / "chunk-000" / "observation.images.depth" / "depth.npy"
        if depth_path.exists():
            depth_array = np.load(depth_path)  # (T, H, W) or (T, H, W, 1)
            if len(depth_array.shape) == 4:
                depth_array = depth_array.squeeze(-1)
            if depth_array.shape[1] != self.image_size or depth_array.shape[2] != self.image_size:
                depth_array = self._resize_depth(depth_array, (self.image_size, self.image_size))
            sample['depth_frames'] = depth_array

        # Load instruction from meta
        meta_path = traj_path / "meta" / "episodes.jsonl"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.loads(f.readline())
                sample['instruction_text'] = meta.get('instruction_text', '')

        return sample

    def _load_from_episode(self, episode: Dict) -> Dict:
        """Load data embedded in episode dict."""
        sample = {}

        positions = episode.get('reference_path', [])
        if positions:
            sample['positions'] = np.array(positions)
            sample['num_steps'] = len(positions)

        sample['instruction_text'] = episode.get('instruction_text', '')

        return sample

    def _resize_images(self, images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image array to target size."""
        from PIL import Image
        resized = []
        for img in images:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(target_size, Image.LANCZOS)
            resized.append(np.array(pil_img))
        return np.array(resized)

    def _resize_depth(self, depths: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize depth array to target size."""
        import cv2
        resized = []
        for depth in depths:
            d = cv2.resize(depth, target_size, interpolation=cv2.INTER_LINEAR)
            resized.append(d)
        return np.array(resized)

    def _sample_frames(self, sample: Dict) -> Dict:
        """Sample frames to max_frames."""
        num_frames = len(sample['rgb_frames'])
        if num_frames <= self.max_frames:
            return sample

        # Uniform sampling
        indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)

        sampled = {}
        for key in ['rgb_frames', 'depth_frames', 'positions', 'orientations', 'yaws', 'actions', 'progress']:
            if key in sample and sample[key] is not None:
                sampled[key] = sample[key][indices]
        sampled['num_steps'] = self.max_frames

        return sampled

    def _prepare_tensors(self, sample: Dict) -> Dict:
        """Convert numpy arrays to torch tensors."""
        result = {}

        # RGB frames: (T, H, W, 3) -> (T, 3, H, W)
        if 'rgb_frames' in sample and sample['rgb_frames'] is not None:
            rgb = torch.from_numpy(sample['rgb_frames']).float() / 255.0
            rgb = rgb.permute(0, 3, 1, 2)  # (T, H, W, 3) -> (T, 3, H, W)
            if self.transform_rgb:
                # Normalize with ImageNet stats
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                rgb = (rgb - mean) / std
            result['rgb'] = rgb

        # Depth frames: (T, H, W) -> (T, 1, H, W)
        if 'depth_frames' in sample and sample['depth_frames'] is not None:
            depth = torch.from_numpy(sample['depth_frames']).float().unsqueeze(1)
            # Normalize depth to [0, 1]
            depth = depth.clamp(0, 10) / 10.0
            result['depth'] = depth

        # Positions: (T, 3)
        if 'positions' in sample and sample['positions'] is not None:
            result['positions'] = torch.from_numpy(sample['positions']).float()

        # Yaws: (T,)
        if 'yaws' in sample and sample['yaws'] is not None:
            result['yaws'] = torch.from_numpy(sample['yaws']).float()

        # Actions: (T,)
        if 'actions' in sample and sample['actions'] is not None:
            result['actions'] = torch.from_numpy(sample['actions']).long()

        # Progress: (T,)
        if 'progress' in sample and sample['progress'] is not None:
            result['progress'] = torch.from_numpy(sample['progress']).float()

        result['num_steps'] = sample.get('num_steps', 0)
        result['instruction_text'] = sample.get('instruction_text', '')

        return result

    def load_clip_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Load CLIP features for frames."""
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device='cpu')

        # frames: (T, 3, H, W)
        batch_size = 8
        features = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            with torch.no_grad():
                feat = self.clip_model.encode_image(batch)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                features.append(feat)

        return torch.cat(features, dim=0)


class BEVMapGenerator:
    """
    Generates BEV (Bird's Eye View) maps from trajectory data.
    Creates occupancy maps, distance transforms, and semantic maps.
    """

    def __init__(
        self,
        map_size: int = 64,
        resolution: float = 0.25,  # meters per cell
        map_range: float = 16.0   # total map size in meters
    ):
        self.map_size = map_size
        self.resolution = resolution
        self.map_range = map_range

    def generate_bev_from_trajectory(
        self,
        positions: torch.Tensor,
        goal_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate BEV map from trajectory positions.

        Args:
            positions: (T, 3) positions in world coordinates
            goal_position: (3,) goal position

        Returns:
            bev_map: (3, map_size, map_size) BEV map
                Channel 0: occupancy (1=free, 0=blocked)
                Channel 1: distance to goal
                Channel 2: trajectory/progress
        """
        bev = torch.zeros(3, self.map_size, self.map_size)

        # Convert world positions to map coordinates
        map_coords = self._world_to_map(positions[:, :2])  # (T, 2)

        # Mark trajectory cells
        for coord in map_coords:
            x, y = coord
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                bev[0, x, y] = 1.0

        # Mark goal
        goal_map = self._world_to_map(goal_position[:2].unsqueeze(0))[0]
        gx, gy = goal_map
        if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
            # Mark goal area with a circle
            radius = 3
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if dx * dx + dy * dy <= radius * radius:
                            bev[1, nx, ny] = 1.0

        # Distance transform from obstacles (using simple dilation)
        bev[0] = self._compute_distance_transform(bev[0])

        return bev

    def _world_to_map(self, positions: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to map indices."""
        # Center the map at origin
        center = self.map_size // 2
        scale = self.map_size / (2 * self.map_range)

        coords = (positions[:, :2] * scale + center).long()
        coords = torch.clamp(coords, 0, self.map_size - 1)
        return coords

    def _compute_distance_transform(self, occupancy: torch.Tensor) -> torch.Tensor:
        """Compute distance transform from occupied cells."""
        # Simple approximation using multiple erosions
        from scipy.ndimage import distance_transform_edt

        occ_np = (occupancy.numpy() > 0.5).astype(np.float32)
        dist = distance_transform_edt(occ_np)

        # Normalize
        if dist.max() > 0:
            dist = dist / dist.max()

        return torch.from_numpy(dist).float()


def create_vln_dataset(
    data_root: str,
    split: str = "train",
    instruction_type: str = "all",
    **kwargs
) -> InteriorNavDataset:
    """Factory function to create VLN dataset."""
    return InteriorNavDataset(
        data_root=data_root,
        split=split,
        instruction_type=instruction_type,
        **kwargs
    )
