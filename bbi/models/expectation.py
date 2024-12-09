"""Module with expectation model."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bbi.environments import GoRight


class ExpectationModel(GoRight):
    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the GoRight environment.

        Args:
            num_prize_indicators (int): Number of prize indicators.
            env_length (int): Length of the grid.
            status_intensities (List[int]): Possible status intensities.
            has_state_offset (bool): Whether to add noise to observations.
            seed (Optional[int]): Seed for reproducibility.
        """
        super().__init__(
            num_prize_indicators=num_prize_indicators,
            env_length=env_length,
            status_intensities=status_intensities,
            has_state_offset=has_state_offset,
            seed=seed,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to its initial state.

        Args:
            seed (Optional[int]): Seed for reproducibility.
            options (Optional[Dict[str, Any]]): Additional options.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        self.state[1] = 5
        # self.previous_status = 5

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        position, current_status, *prize_indicators = self.state

        direction = 1 if action > 0 else -1
        next_pos = np.clip(position + direction, 0, self.length - 1)

        next_status = 5

        next_prize_indicators = self._compute_next_prize_indicators(
            next_pos, position, next_status, np.array(prize_indicators)
        )

        # Update state
        self.state[0] = next_pos
        self.state[1] = next_status
        self.state[2:] = next_prize_indicators

        reward = self._compute_reward(next_prize_indicators, action, position)
        self.previous_status = None
        self.last_action = action
        self.last_pos = position
        self.last_reward = reward

        # Update cumulative stats
        self.total_reward += reward
        self.action_count += 1

        return self._get_observation(), reward, False, False, {}

    def _compute_next_prize_indicators(
        self,
        next_position: float,
        position: float,
        next_status: int,
        prize_indicators: np.ndarray,
    ) -> np.ndarray:
        """Computes the next prize indicators based on the current state."""
        if int(next_position) == self.length - 1:
            if int(position) == self.length - 2:
                return np.ones_like(prize_indicators, dtype=float) / 3.0
            elif all(prize_indicators == 1):
                return prize_indicators
            else:
                return self._shift_prize_indicators(prize_indicators)
        return np.zeros(self.num_prize_indicators, dtype=int)

    # def _compute_reward(self, next_prize_indicators: np.ndarray, action: int, position: float) -> float:
    #     """Computes the expected reward in the expectation model."""
    #     return 0.0 if action == 0 else -1.0
