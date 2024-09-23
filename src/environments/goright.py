# goright.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

STATUS_TABLE = {
    (0, 0): 5,
    (0, 5): 0,
    (0, 10): 5,
    (5, 0): 10,
    (5, 5): 10,
    (5, 10): 10,
    (10, 0): 0,
    (10, 5): 5,
    (10, 10): 0
}

class GoRightEnv(gym.Env):
    """Custom Gymnasium environment for the "Go Right" task."""

    def __init__(
        self,
        num_prize_indicators: int = 2,
        length: int = 10,
        status_intensities: List[int] = [0, 5, 10],
        is_observation_noisy: bool = False,
        seed: Optional[int] = None
    ) -> None:
        """Initializes the GoRight environment."""
        super().__init__()
        self.num_prize_indicators = num_prize_indicators
        self.length = length
        self.max_intensity = max(status_intensities)
        self.previous_status: Optional[int] = None
        self.is_observation_noisy = is_observation_noisy

        if seed is not None:
            np.random.seed(seed)

        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Discrete(length),
                "intensity": spaces.Box(low=np.array([0]), high=np.array([10]), shape=(1,), dtype=np.int8),
                "prize_status": spaces.Box(low=np.array([0]*num_prize_indicators), high=np.array([1]*num_prize_indicators), shape=(num_prize_indicators,), dtype=np.int8),
            }
        )

        self.state: np.ndarray = np.array([0] + [0] + [0] * self.num_prize_indicators, dtype=int)
        self.is_right_low_intensity: bool = False

        self.position_offset: float = 0.0
        self.status_indicator_offset: float = 0.0
        self.prize_indicator_offset: float = 0.0

        self.reset(seed=seed)

    def reset(self, seed: Optional[int]) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to its initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.state = np.array([0] + [0] + [0] * self.num_prize_indicators, dtype=int)
        self.is_right_low_intensity = False
        self.previous_status = 0

        if self.is_observation_noisy:
            self.position_offset = np.random.uniform(-0.25, 0.25)
            self.status_indicator_offset = np.random.uniform(-1.25, 1.25)
            self.prize_indicator_offset = np.random.uniform(-0.25, 0.25)
        else:
            self.position_offset = 0.0
            self.status_indicator_offset = 0.0
            self.prize_indicator_offset = 0.0

        return self._add_noise_to_state(self.state.copy()), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment given an action."""
        position, current_status, *prize_indicators = self.state

        # Determine the next position and status
        direction = 1 if action > 0 else -1
        next_pos = max(0, min(self.length - 1, position + direction))  # Constrain within bounds

        next_status = STATUS_TABLE.get((self.previous_status, current_status), 0)
        next_prize_indicators = self._compute_next_prize_indicators(next_pos, position, next_status, prize_indicators)

        # Update the state
        self.state[0] = next_pos
        self.state[1] = next_status
        self.state[2:] = next_prize_indicators

        reward = self._compute_reward(next_prize_indicators, action)
        self.previous_status = current_status

        # Return the noisy state
        return self._add_noise_to_state(self.state.copy()), reward, False, False, {}

    def _compute_next_prize_indicators(self, next_position, position, next_status, prize_indicators):
        if next_position != self.length - 1:
            self.is_right_low_intensity = False
            return np.zeros(self.num_prize_indicators, dtype=int)
        elif all(prize_indicators) == 1:
            self.is_right_low_intensity = False
            return np.ones(self.num_prize_indicators, dtype=int)
        elif position == self.length - 2:
            if next_status == self.max_intensity:
                self.is_right_low_intensity = False
                return np.ones(self.num_prize_indicators, dtype=int)

        if self.is_right_low_intensity:
            return self._shift_prize_indicators(prize_indicators)
        else:
            self.is_right_low_intensity = True
            return np.zeros(self.num_prize_indicators, dtype=int)

    def _shift_prize_indicators(self, prize_indicators):
        prize_indicators = list(prize_indicators)
        if 1 not in prize_indicators:
            prize_indicators[0] = 1
        else:
            index = prize_indicators.index(1)
            if index == self.num_prize_indicators - 1:
                prize_indicators = [0] * self.num_prize_indicators
            else:
                prize_indicators[index] = 0
                prize_indicators[index + 1] = 1
        return np.array(prize_indicators, dtype=int)

    def _compute_reward(self, next_prize_indicators, action):
        if all(next_prize_indicators) == 1:
            return 3
        if action == 0:  # left
            return 0
        return -1  # right

    def _add_noise_to_state(self, state):
        if self.is_observation_noisy:
            state[0] = state[0] + self.position_offset
            state[1] = state[1] + self.status_indicator_offset
            state[2:] = state[2:] + self.prize_indicator_offset
        return state
