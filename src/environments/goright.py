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
        length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = False,
        seed: Optional[int] = None
    ) -> None:
        """Initializes the GoRight environment."""
        super().__init__()
        self.num_prize_indicators = num_prize_indicators
        self.length = length
        self.max_intensity = max(status_intensities)
        self.intensities = status_intensities
        self.previous_status: Optional[int] = None
        self.has_state_offset = has_state_offset

        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Discrete(length),
                "intensity": spaces.Discrete(len(status_intensities)),
                "prize_status": spaces.Box(low=np.zeros(num_prize_indicators), high=np.ones(num_prize_indicators), shape=(num_prize_indicators,), dtype=np.int8),
            }
        )
        
        self.reset(seed=seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(int(np.random.random(1)[0]*100 if seed is None else seed))
        return [seed]

    def reset(self, seed: Optional[int]=None, new_state: Optional[np.ndarray]=None, previous_status: Optional[int]=None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to its initial state."""
        self.seed(seed)

        if new_state is not None:
            self.state: np.ndarray = new_state
            self.previous_status: int = previous_status
        else:
            self.state: np.ndarray = np.zeros(2+self.num_prize_indicators, dtype=float)
            self.state[1] = self.np_random.choice(self.intensities)
            self.previous_status: int = self.np_random.choice(self.intensities)

        if self.has_state_offset:
            self.position_offset = self.np_random.uniform(-0.25, 0.25)
            self.status_indicator_offset = self.np_random.uniform(-1.25, 1.25)
            self.prize_indicator_offsets = self.np_random.uniform(-0.25, 0.25, size=self.num_prize_indicators)

        return self._add_noise_to_state(self.state.copy()), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment given an action."""
        position, current_status, *prize_indicators = self.state

        # Determine the next position and status
        direction = 1 if action > 0 else -1
        next_pos = np.clip(position + direction, a_min=0, a_max=self.length - 1)  # Constrain within bounds

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
        if next_position == self.length - 1: # if I am going to position 10
            if position == self.length - 2: # if I am at position 9
                if next_status == self.max_intensity: # if intensity = 10
                    return np.ones(self.num_prize_indicators, dtype=int) # 11  
            else: # I am at position 10
                if all(prize_indicators) == 1: # receiving prize
                    return prize_indicators
                else: # not receiving prize
                    return self._shift_prize_indicators(prize_indicators)
        
        return np.zeros(self.num_prize_indicators, dtype=int)

    def _shift_prize_indicators(self, prize_indicators):
        if np.sum(prize_indicators) == 0: # all zeros
            prize_indicators[0] = 1
        else: # has a 1 in it, shift it
            one_index = np.argmax(prize_indicators)
            if one_index == self.num_prize_indicators - 1: # if one in the last place, return all zeros
                prize_indicators[one_index] = 0
            else:
                prize_indicators[one_index] = 0
                prize_indicators[one_index + 1] = 1

        return prize_indicators

    def _compute_reward(self, next_prize_indicators, action):
        if all(next_prize_indicators) == 1:
            return 3
        if action == 0:  # left
            return 0
        return -1  # right

    def _add_noise_to_state(self, state):
        if self.has_state_offset:
            state[0] = state[0] + self.position_offset
            state[1] = state[1] + self.status_indicator_offset
            state[2:] = state[2:] + self.prize_indicator_offsets
        return state
