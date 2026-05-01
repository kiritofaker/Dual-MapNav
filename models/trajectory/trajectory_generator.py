"""Trajectory generation via video interpolation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


def video_interpolate(
    frames: torch.Tensor,
    num_interpolated: int = 8
) -> torch.Tensor:
    """
    Interpolate between video frames for smoother trajectories.

    Args:
        frames: (B, C, T, H, W) video frames
        num_interpolated: Number of frames to interpolate between each pair

    Returns:
        interpolated: (B, C, T * (num_interpolated + 1), H, W) interpolated video
    """
    B, C, T, H, W = frames.shape

    # Create output
    total_frames = T * (num_interpolated + 1)
    interpolated = torch.zeros(B, C, total_frames, H, W, device=frames.device)

    for t in range(T - 1):
        start_frame = frames[:, :, t]
        end_frame = frames[:, :, t + 1]

        # Store start frame
        out_idx = t * (num_interpolated + 1)
        interpolated[:, :, out_idx] = start_frame

        # Interpolate frames
        for i in range(num_interpolated):
            alpha = (i + 1) / (num_interpolated + 1)
            interp_frame = (1 - alpha) * start_frame + alpha * end_frame
            out_idx = t * (num_interpolated + 1) + i + 1
            interpolated[:, :, out_idx] = interp_frame

    # Store last frame
    interpolated[:, :, -1] = frames[:, :, -1]

    return interpolated


class TrajectoryGenerator(nn.Module):
    """
    Generates navigation trajectories from predicted maps.
    Uses video interpolation for smooth path generation.
    """

    def __init__(
        self,
        map_size: int = 64,
        num_keyframes: int = 16,
        interpolation_factor: int = 4
    ):
        super().__init__()

        self.map_size = map_size
        self.num_keyframes = num_keyframes
        self.interpolation_factor = interpolation_factor

    def generate_trajectory_from_maps(
        self,
        map_predictions: torch.Tensor,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Generate a navigation trajectory from predicted maps.

        Args:
            map_predictions: (B, C, T, H, W) predicted map sequence
            start_pos: Starting (x, y) position
            goal_pos: Goal (x, y) position

        Returns:
            interpolated_trajectory: (B, C, T*interp, H, W) smooth trajectory
            path: List of (x, y) waypoints
        """
        B, C, T, H, W = map_predictions.shape

        # Extract navigable paths from each map
        paths = []
        for t in range(T):
            map_t = map_predictions[:, :, t]
            path_t = self._extract_path(map_t, start_pos if t == 0 else paths[-1][-1], goal_pos)
            paths.append(path_t)

        # Concatenate paths
        full_path = []
        for path in paths:
            full_path.extend(path)

        # Convert path to trajectory tensor
        trajectory = self._path_to_trajectory(full_path, map_predictions.shape)

        # Interpolate for smoothness
        if self.interpolation_factor > 1:
            trajectory = video_interpolate(trajectory, self.interpolation_factor)

        return trajectory, full_path

    def _extract_path(
        self,
        map_pred: torch.Tensor,
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Extract a navigable path from a map prediction.

        Uses simple A* or gradient descent towards goal.
        """
        navigable = map_pred[0] > 0.5 if map_pred.shape[0] > 0 else torch.ones_like(map_pred[0])

        # Simple greedy path
        path = [start_pos]
        current = start_pos

        for _ in range(100):  # Max steps
            if current == goal_pos:
                break

            # Direction towards goal
            dx = np.sign(goal_pos[0] - current[0])
            dy = np.sign(goal_pos[1] - current[1])

            # Try to move towards goal
            next_pos = current
            if dx != 0:
                candidate = (current[0] + dx, current[1])
                if navigable[candidate[0], candidate[1]] > 0:
                    next_pos = candidate
            if dy != 0:
                candidate = (current[0], current[1] + dy)
                if navigable[candidate[0], candidate[1]] > 0:
                    next_pos = candidate

            if next_pos == current:
                break

            path.append(next_pos)
            current = next_pos

        return path

    def _path_to_trajectory(
        self,
        path: List[Tuple[int, int]],
        target_shape: torch.Size
    ) -> torch.Tensor:
        """Convert path to trajectory tensor."""
        B, C, T, H, W = target_shape

        trajectory = torch.zeros(B, C, len(path), H, W, device=next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu')

        for t, (x, y) in enumerate(path):
            if 0 <= x < H and 0 <= y < W:
                trajectory[:, :, t, x, y] = 1.0

        return trajectory

    def smooth_trajectory(
        self,
        trajectory: torch.Tensor,
        method: str = "gaussian"
    ) -> torch.Tensor:
        """
        Apply smoothing to trajectory.

        Args:
            trajectory: (B, C, T, H, W) trajectory tensor
            method: Smoothing method ("gaussian", "median", "bilateral")

        Returns:
            smoothed: Smoothed trajectory
        """
        if method == "gaussian":
            # Apply Gaussian smoothing along time dimension
            kernel_size = 5
            sigma = 1.0
            kernel = self._gaussian_kernel(kernel_size, sigma)
            kernel = kernel.view(1, 1, -1, 1, 1).to(trajectory.device)

            # Pad along time dimension
            pad = kernel_size // 2
            padded = F.pad(trajectory, (0, 0, 0, 0, pad, pad), mode='replicate')

            # Convolve
            smoothed = F.conv3d(padded, kernel)
            return smoothed

        elif method == "median":
            # Median filter along time
            from scipy.ndimage import median_filter
            trajectory_np = trajectory.cpu().numpy()
            smoothed_np = median_filter(trajectory_np, size=(1, 1, 3, 1, 1))
            return torch.from_numpy(smoothed_np).to(trajectory.device)

        return trajectory

    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        x = torch.arange(size, dtype=torch.float32) - size // 2
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel


class PathPlanner:
    """
    A* based path planner for navigation.
    """

    def __init__(
        self,
        map_size: int = 64,
        num_actions: int = 4
    ):
        self.map_size = map_size
        self.num_actions = num_actions
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W

    def plan(
        self,
        costmap: torch.Tensor,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Plan path using A* algorithm.

        Args:
            costmap: (H, W) navigability cost
            start: Starting position
            goal: Goal position

        Returns:
            path: List of (x, y) positions
        """
        import heapq

        # Ensure tensors are on CPU for A*
        costmap = costmap.cpu().numpy()
        H, W = costmap.shape

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(heuristic(start, goal), counter, start)]
        heapq.heapify(open_set)

        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if 0 <= neighbor[0] < H and 0 <= neighbor[1] < W:
                    # Cost is inverse of navigability
                    move_cost = 1.0 - costmap[neighbor[0], neighbor[1]]
                    tentative_g = g_score[current] + move_cost

                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        counter += 1
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

        return [start]  # No path found, return start

    def smooth_path(
        self,
        path: List[Tuple[int, int]],
        costmap: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Apply path smoothing to reduce turns.

        Args:
            path: Original path
            costmap: Navigability costmap

        Returns:
            smoothed_path: Path with reduced waypoints
        """
        if len(path) <= 2:
            return path

        costmap = costmap.cpu().numpy()
        smoothed = [path[0]]

        i = 0
        while i < len(path) - 1:
            # Try to skip waypoints
            best_j = i + 1

            for j in range(len(path) - 1, i, -1):
                if self._is_line_clear(path[i], path[j], costmap):
                    best_j = j
                    break

            smoothed.append(path[best_j])
            i = best_j

        return smoothed

    def _is_line_clear(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        costmap: np.ndarray
    ) -> bool:
        """Check if direct line between two points is clear."""
        x0, y0 = start
        x1, y1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if costmap[x0, y0] < 0.3:  # Blocked
                return False

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return True
