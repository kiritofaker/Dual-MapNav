"""Logging utilities."""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "Dual-MapNav",
    log_dir: str = "./logs",
    level: int = logging.INFO,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name

    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # WandB
    if use_wandb and wandb_project:
        try:
            import wandb
            wandb.init(project=wandb_project, name=name)
            logger.info(f"Initialized W&B project: {wandb_project}")
        except ImportError:
            logger.warning("wandb not installed, skipping")

    return logger


class MetricLogger:
    """
    Metric logger for tracking training progress.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        self.metrics = {}
        self.history = {}
        os.makedirs(log_dir, exist_ok=True)

    def log(self, metrics: dict, step: int):
        """Log metrics at given step."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))

        self.metrics = metrics

    def save(self, filename: str = "metrics.json"):
        """Save metrics history to file."""
        import json
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_latest(self, key: str) -> float:
        """Get latest value for a metric."""
        if key in self.history and len(self.history[key]) > 0:
            return self.history[key][-1][1]
        return 0.0

    def get_average(self, key: str, last_n: int = 100) -> float:
        """Get average of last N values for a metric."""
        if key not in self.history:
            return 0.0
        values = [v for _, v in self.history[key][-last_n:]]
        return sum(values) / len(values) if values else 0.0
