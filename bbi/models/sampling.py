"""Module with sampling model."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bbi.environments import GoRight


class SamplingModel(GoRight):
    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = False,
        seed: Optional[int] = None,
    ) -> None:
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
        super().reset(seed=seed)
        self.state[1] = np.random.choice(self.intensities)
        # self.previous_status = np.random.choice(self.intensities)

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        position, current_status, *prize_indicators = self.state

        direction = 1 if action > 0 else -1
        next_pos = np.clip(position + direction, 0, self.length - 1)

        next_status = np.random.choice(self.intensities)
        next_prize_indicators = self._compute_next_prize_indicators(
            next_pos, position, next_status, np.array(prize_indicators)
        )

        # Update state
        self.state[0] = next_pos
        self.state[1] = next_status
        self.state[2:] = next_prize_indicators

        reward = self._compute_reward(next_prize_indicators, action, position)
        # self.previous_status = current_status

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
                return np.random.choice(
                    [0, 1], size=self.num_prize_indicators, p=[2 / 3, 1 / 3]
                )
            elif all(prize_indicators == 1):
                return prize_indicators
            else:
                return self._shift_prize_indicators(prize_indicators)
        return np.zeros(self.num_prize_indicators, dtype=int)
