"""Module with BBI model."""

from typing import List, Optional

import numpy as np

from bbi.environments import GoRight


class BBI(GoRight):
    metadata = {
        "render_modes": ["human"],
        "environment_name": "1-step Predicted Variance Model",
    }

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
            has_state_offset=False,
            seed=seed,
        )

    def step(self, action):
        if self.state.ndim == 1:
            position, current_status, *prize_indicators = self.state
            prize_indicators_min = np.array(prize_indicators)
            prize_indicators_max = prize_indicators_min
        else:
            position_min = self.state[0, 0]
            # position_max = self.state[1, 0]
            # status_min_current = self.state[0, 1]
            # status_max_current = self.state[1, 1]
            position = position_min
            prize_indicators_min = self.state[0, 2:]
            prize_indicators_max = self.state[1, 2:]

        direction = 1 if action > 0 else -1
        next_pos = np.clip(position + direction, 0, self.length - 1)

        status_min = np.min(self.intensities)
        status_max = np.max(self.intensities)

        prize_indicators_min = self._compute_next_prize_indicators(
            next_pos, position, status_min, prize_indicators_min
        )
        prize_indicators_max = self._compute_next_prize_indicators(
            next_pos, position, status_max, prize_indicators_max
        )

        reward_min = self._compute_reward(prize_indicators_min, action, position)
        reward_max = self._compute_reward(prize_indicators_max, action, position)
        self.previous_status = None

        state_bounding_box = np.zeros((2, 2 + self.num_prize_indicators))
        state_bounding_box[0, 0] = next_pos
        state_bounding_box[1, 0] = next_pos
        state_bounding_box[0, 1] = status_min
        state_bounding_box[1, 1] = status_max
        for i in range(self.num_prize_indicators):
            p_min = prize_indicators_min[i]
            p_max = prize_indicators_max[i]
            state_bounding_box[0, 2 + i] = p_min
            state_bounding_box[1, 2 + i] = p_max

        self.state = state_bounding_box

        return state_bounding_box, np.array([reward_min, reward_max]), False, False, {}
