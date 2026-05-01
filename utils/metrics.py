"""Evaluation metrics for VLN."""

import torch
import numpy as np
from typing import List, Dict, Tuple
import numpy as np


def compute_vln_metrics(
    predicted_trajectory: List[Tuple[int, int]],
    ground_truth_trajectory: List[Tuple[int, int]],
    goal_position: Tuple[int, int],
    start_position: Tuple[int, int],
    success_threshold: float = 3.0
) -> Dict[str, float]:
    """
    Compute VLN evaluation metrics.

    Args:
        predicted_trajectory: List of (x, y) positions
        ground_truth_trajectory: Ground truth path
        goal_position: Goal (x, y)
        start_position: Start (x, y)
        success_threshold: Distance threshold for success

    Returns:
        Dictionary with metrics
    """
    # Success
    final_position = predicted_trajectory[-1] if predicted_trajectory else start_position
    distance_to_goal = np.sqrt(
        (final_position[0] - goal_position[0])**2 +
        (final_position[1] - goal_position[1])**2
    )
    success = distance_to_goal < success_threshold

    # Path length
    path_length = len(predicted_trajectory)

    # Trajectory length (Euclidean)
    traj_length = 0.0
    for i in range(1, len(predicted_trajectory)):
        dx = predicted_trajectory[i][0] - predicted_trajectory[i-1][0]
        dy = predicted_trajectory[i][1] - predicted_trajectory[i-1][1]
        traj_length += np.sqrt(dx**2 + dy**2)

    # Oracle path length (shortest path to goal)
    oracle_length = np.sqrt(
        (goal_position[0] - start_position[0])**2 +
        (goal_position[1] - start_position[1])**2
    )

    # SPL (Success weighted by Path Length)
    spl = traj_length / max(oracle_length, 1e-6) if success else 0.0

    # NE (Navigation Error)
    min_distance = float('inf')
    for pos in predicted_trajectory:
        dist = np.sqrt((pos[0] - goal_position[0])**2 + (pos[1] - goal_position[1])**2)
        min_distance = min(min_distance, dist)

    navigation_error = min_distance

    return {
        'success': float(success),
        'path_length': path_length,
        'trajectory_length': traj_length,
        'oracle_length': oracle_length,
        'spl': spl,
        'navigation_error': navigation_error
    }


def compute_map_metrics(
    predicted_map: torch.Tensor,
    ground_truth_map: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for map prediction accuracy.

    Args:
        predicted_map: (C, H, W) or (B, C, H, W) predicted map
        ground_truth_map: (C, H, W) or (B, C, H, W) ground truth

    Returns:
        Dictionary with metrics
    """
    if predicted_map.dim() == 4:
        predicted_map = predicted_map[0]
    if ground_truth_map.dim() == 4:
        ground_truth_map = ground_truth_map[0]

    # MSE
    mse = torch.mean((predicted_map - ground_truth_map) ** 2).item()

    # MAE
    mae = torch.mean(torch.abs(predicted_map - ground_truth_map)).item()

    # Correlation
    pred_flat = predicted_map.flatten().cpu().numpy()
    gt_flat = ground_truth_map.flatten().cpu().numpy()
    correlation = np.corrcoef(pred_flat, gt_flat)[0, 1] if len(pred_flat) > 1 else 0.0

    # IoU for traversability (channel 0)
    pred_trav = (predicted_map[0] > 0.5).float()
    gt_trav = (ground_truth_map[0] > 0.5).float()
    intersection = (pred_trav * gt_trav).sum()
    union = ((pred_trav + gt_trav) > 0).float().sum()
    iou = (intersection / (union + 1e-6)).item()

    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'iou': iou
    }


def compute_trajectory_metrics(
    predicted_trajectory: torch.Tensor,
    ground_truth_trajectory: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for trajectory prediction.

    Args:
        predicted_trajectory: (T, 2) predicted path
        ground_truth_trajectory: (T, 2) ground truth path

    Returns:
        Dictionary with metrics
    """
    pred = predicted_trajectory.cpu().numpy()
    gt = ground_truth_trajectory.cpu().numpy()

    # MSE
    mse = np.mean((pred - gt) ** 2)

    # MAE
    mae = np.mean(np.abs(pred - gt))

    # DTW (Dynamic Time Warping) distance
    dtw_distance = compute_dtw(pred, gt)

    return {
        'mse': mse,
        'mae': mae,
        'dtw_distance': dtw_distance
    }


def compute_dtw(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.

    Args:
        p: (T1, D) first sequence
        q: (T2, D) second sequence

    Returns:
        DTW distance
    """
    T1, T2 = len(p), len(q)

    # Build cost matrix
    dtw_matrix = np.full((T1 + 1, T2 + 1), float('inf'))
    dtw_matrix[0, 0] = 0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = np.linalg.norm(p[i-1] - q[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]  # match
            )

    return dtw_matrix[T1, T2]
