"""Default configuration for Dual-MapNav."""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
import os


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Map representation
    bev_channels: int = 64
    semantic_classes: int = 10
    embedding_dim: int = 512
    map_size: int = 64
    num_frames: int = 16

    # Observation encoder
    rgb_backbone: str = "resnet50"
    rgb_channels: int = 2048
    depth_channels: int = 512
    fusion_embed_dim: int = 768
    num_heads: int = 8

    # Diffusion model
    diffusion_channels: int = 16
    time_embed_dim: int = 256
    condition_dim: int = 768
    num_timesteps: int = 1000
    guidance_scale: float = 7.5

    # UNet3D
    base_channels: int = 128
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    attention_resolutions: List[int] = field(default_factory=lambda: [4, 2, 1])


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999

    # Training stages
    stage1_epochs: int = 100
    stage2_epochs: int = 50
    batch_size: int = 8
    gradient_accumulation_steps: int = 1

    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 1000

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 10
    resume_from: Optional[str] = None

    # Logging
    log_frequency: int = 100
    eval_frequency: int = 1000
    project_name: str = "Dual-MapNav"


@dataclass
class DataConfig:
    """Data configuration."""
    # Synthetic data
    use_synthetic: bool = True
    synthetic_data_size: int = 1000
    synthetic_map_size: int = 64

    # Real datasets (for future)
    dataset_name: str = "synthetic"
    data_root: str = "./data"
    num_workers: int = 4


@dataclass
class DefaultConfig:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment
    seed: int = 42
    device: str = "cuda"
    debug: bool = False


def get_config(config_path: Optional[str] = None) -> DefaultConfig:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        # Convert dict to nested config objects
        return DefaultConfig(
            model=ModelConfig(**cfg_dict.get('model', {})),
            training=TrainingConfig(**cfg_dict.get('training', {})),
            data=DataConfig(**cfg_dict.get('data', {})),
            seed=cfg_dict.get('seed', 42),
            device=cfg_dict.get('device', 'cuda'),
            debug=cfg_dict.get('debug', False)
        )
    return DefaultConfig()
