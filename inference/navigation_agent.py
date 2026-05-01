"""Navigation agent using Dual-MapNav for map-based VLN."""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import numpy as np

from models.map_predictor import Dual-MapNavDiffusion
from inference.map_inference import MapInference


class Dual-MapNavAgent:
    """
    Navigation agent that uses Dual-MapNav for map-based planning.

    The agent:
    1. Maintains a belief map of the environment
    2. Uses Dual-MapNav to predict future maps given instructions
    3. Plans paths based on predicted maps
    4. Executes navigation actions
    """

    def __init__(
        self,
        model: Dual-MapNavDiffusion,
        device: str = "cuda",
        map_size: int = 64,
        num_actions: int = 4
    ):
        """
        Args:
            model: Dual-MapNavDiffusion model
            device: Device to run on
            map_size: Size of the map grid
            num_actions: Number of navigation actions (e.g., 4 for cardinal directions)
        """
        self.model = model
        self.device = device
        self.map_size = map_size
        self.num_actions = num_actions

        # Move model to device and eval mode
        self.model.to(device)
        self.model.eval()

        # Create inference wrapper
        self.inference = MapInference(
            model=model,
            device=device
        )

        # Agent state
        self.current_position = (map_size // 2, map_size // 2)  # Center of map
        self.current_heading = 0  # 0 = North, increases clockwise
        self.belief_map = None
        self.visited_cells = set()
        self.trajectory = []

    def reset(self, start_position: Tuple[int, int]):
        """Reset agent state for new episode."""
        self.current_position = start_position
        self.current_heading = 0
        self.belief_map = torch.zeros(3, self.map_size, self.map_size, device=self.device)
        self.belief_map[:, start_position[0], start_position[1]] = 1.0
        self.visited_cells = {start_position}
        self.trajectory = [start_position]

    def step(
        self,
        instruction: str,
        observation: Dict[str, torch.Tensor]
    ) -> Tuple[int, float]:
        """
        Take a navigation step.

        Args:
            instruction: Text instruction (e.g., "Go to the kitchen")
            observation: Dictionary with RGB, depth, etc.

        Returns:
            action: Action index
            reward: Reward for this step
        """
        rgb = observation.get('rgb')
        depth = observation.get('depth')

        # Update belief map with observation
        self._update_belief(rgb, depth)

        # Predict future maps using Dual-MapNav
        predicted_maps = self._predict_future_maps(instruction, observation)

        # Plan path based on predictions
        action = self._plan_action(predicted_maps)

        # Execute action
        self._execute_action(action)

        # Compute reward
        reward = self._compute_reward(predicted_maps, action)

        return action, reward

    def _update_belief(
        self,
        rgb: Optional[torch.Tensor],
        depth: Optional[torch.Tensor]
    ):
        """Update the agent's belief map with new observation."""
        x, y = self.current_position

        # Mark current position as visited
        self.belief_map[:, x, y] = 0.0  # Clear current

        # Simple update: add observation influence around current position
        radius = 3
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= radius:
                        self.belief_map[0, nx, ny] = max(
                            self.belief_map[0, nx, ny],
                            1.0 - dist / radius
                        )

    def _predict_future_maps(
        self,
        instruction: str,
        observation: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Use Dual-MapNav to predict future maps."""
        with torch.no_grad():
            # Create a partial map from belief
            bev_map = self.belief_map.unsqueeze(0)

            # Use observation features
            rgb = observation.get('rgb')
            depth = observation.get('depth')

            # Category IDs (placeholder)
            category_ids = torch.zeros(1, 10, device=self.device, dtype=torch.long)

            # Generate map prediction
            predicted = self.inference.generate_map(
                bev_map=bev_map,
                semantic_map=torch.zeros(1, 10, self.map_size, self.map_size, device=self.device),
                category_ids=category_ids,
                rgb=rgb,
                depth=depth,
                use_cfg=True
            )

        return predicted

    def _plan_action(self, predicted_maps: torch.Tensor) -> int:
        """
        Plan next action based on predicted maps.

        Args:
            predicted_maps: (1, C, 1, H, W) predicted map

        Returns:
            action: Action index (0=forward, 1=right, 2=back, 3=left)
        """
        # Simple planning: go towards unexplored areas
        navigable = predicted_maps.squeeze(2)[0]  # Channel 0 = traversability

        # Check in front of agent
        directions = [
            (0, 1),   # Forward (up)
            (1, 0),   # Right
            (0, -1),  # Back
            (-1, 0)   # Left
        ]

        best_action = 0
        best_score = -float('inf')

        x, y = self.current_position

        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                score = navigable[nx, ny].item()
                # Penalize visited cells
                if (nx, ny) in self.visited_cells:
                    score -= 0.5
                if score > best_score:
                    best_score = score
                    best_action = i

        return best_action

    def _execute_action(self, action: int):
        """Execute the chosen action."""
        # Action mapping: 0=forward, 1=right, 2=back, 3=left
        direction_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        dx, dy = direction_vectors[action]
        x, y = self.current_position
        nx, ny = x + dx, y + dy

        # Clamp to map bounds
        nx = max(0, min(self.map_size - 1, nx))
        ny = max(0, min(self.map_size - 1, ny))

        self.current_position = (nx, ny)
        self.visited_cells.add((nx, ny))
        self.trajectory.append((nx, ny))

        # Update heading
        self.current_heading = (self.current_heading + action) % 4

    def _compute_reward(
        self,
        predicted_maps: torch.Tensor,
        action: int
    ) -> float:
        """Compute reward for the taken action."""
        # Positive reward for reaching new cells
        if self.current_position not in self.visited_cells:
            return 1.0

        # Small negative reward for step
        return -0.01

    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Get the agent's trajectory so far."""
        return self.trajectory.copy()

    def get_map(self) -> torch.Tensor:
        """Get current belief map."""
        return self.belief_map.clone()


class VLNEvaluator:
    """
    Evaluator for VLN tasks using Dual-MapNav agent.
    """

    def __init__(
        self,
        agent: Dual-MapNavAgent,
        success_threshold: float = 3.0
    ):
        self.agent = agent
        self.success_threshold = success_threshold

    def evaluate_episode(
        self,
        start_position: Tuple[int, int],
        goal_position: Tuple[int, int],
        instruction: str,
        max_steps: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate a single VLN episode.

        Args:
            start_position: Starting (x, y) position
            goal_position: Goal (x, y) position
            instruction: Navigation instruction
            max_steps: Maximum number of steps

        Returns:
            metrics: Dictionary with evaluation metrics
        """
        # Reset agent
        self.agent.reset(start_position)

        episode_reward = 0.0
        steps = 0
        success = False

        for step in range(max_steps):
            # Get observation (placeholder)
            observation = {
                'rgb': torch.randn(1, 3, 224, 224, device=self.agent.device),
                'depth': torch.rand(1, 1, 224, 224, device=self.agent.device)
            }

            # Take step
            action, reward = self.agent.step(instruction, observation)
            episode_reward += reward
            steps += 1

            # Check if reached goal
            distance_to_goal = np.sqrt(
                (self.agent.current_position[0] - goal_position[0])**2 +
                (self.agent.current_position[1] - goal_position[1])**2
            )

            if distance_to_goal < self.success_threshold:
                success = True
                break

        # Compute metrics
        trajectory = self.agent.get_trajectory()
        path_length = len(trajectory)
        trajectory_length = sum(
            np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 +
                    (trajectory[i][1] - trajectory[i-1][1])**2)
            for i in range(1, len(trajectory))
        )

        metrics = {
            'success': float(success),
            'steps': steps,
            'path_length': path_length,
            'trajectory_length': trajectory_length,
            'reward': episode_reward,
            'spl': trajectory_length / max(1, abs(start_position[0] - goal_position[0]) +
                                           abs(start_position[1] - goal_position[1])) if success else 0.0
        }

        return metrics

    def evaluate_dataset(
        self,
        episodes: List[Dict],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on a dataset of episodes.

        Args:
            episodes: List of episode dictionaries
            verbose: Whether to print progress

        Returns:
            results: Aggregated evaluation metrics
        """
        all_metrics = []

        for i, episode in enumerate(episodes):
            metrics = self.evaluate_episode(
                start_position=episode['start'],
                goal_position=episode['goal'],
                instruction=episode['instruction']
            )
            all_metrics.append(metrics)

            if verbose and (i + 1) % 10 == 0:
                print(f"Evaluated {i+1}/{len(episodes)} episodes")

        # Aggregate metrics
        results = {
            'success_rate': np.mean([m['success'] for m in all_metrics]),
            'avg_steps': np.mean([m['steps'] for m in all_metrics]),
            'avg_path_length': np.mean([m['path_length'] for m in all_metrics]),
            'avg_trajectory_length': np.mean([m['trajectory_length'] for m in all_metrics]),
            'spl': np.mean([m['spl'] for m in all_metrics]),
        }

        return results
