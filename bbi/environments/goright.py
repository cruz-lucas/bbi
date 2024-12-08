"""Module defining the GoRight Gymnasium environment."""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

STATUS_TABLE = {
    (0, 0): 5,
    (0, 5): 0,
    (0, 10): 5,
    (5, 0): 10,
    (5, 5): 10,
    (5, 10): 10,
    (10, 0): 0,
    (10, 5): 5,
    (10, 10): 0,
}


class GoRight(gym.Env):
    """Custom Gymnasium environment for the "Go Right" task.

    The agent moves along a 1D grid, aiming to collect prizes at the end.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = True,
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
        super().__init__()
        self.num_prize_indicators = num_prize_indicators
        self.length = env_length
        self.max_intensity = max(status_intensities)
        self.intensities = status_intensities
        self.previous_status: Optional[int] = None
        self.has_state_offset = has_state_offset
        self.render_mode = None

        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Box(
            low=0.0,
            high=max(env_length - 1, self.max_intensity, 1.0),
            shape=(2 + num_prize_indicators,),
            dtype=np.float32,
        )

        self.state: Optional[np.ndarray] = None
        self.previous_status: Optional[int] = None

        self.seed(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        """Sets the seed for the environment's random number generator."""
        self.np_random, _ = gym.utils.seeding.np_random(seed)

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
        self.state = np.zeros(2 + self.num_prize_indicators, dtype=np.float32)
        self.state[1] = self.np_random.choice(self.intensities)
        self.previous_status = self.np_random.choice(self.intensities)

        if self.has_state_offset:
            self.position_offset = self.np_random.uniform(-0.25, 0.25)
            self.status_indicator_offset = self.np_random.uniform(-1.25, 1.25)
            self.prize_indicator_offsets = self.np_random.uniform(
                -0.25, 0.25, size=self.num_prize_indicators
            )

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one time step within the environment.

        Args:
            action (int): The action taken by the agent.

        Returns:
            Tuple containing:
                - observation (np.ndarray): The next observation.
                - reward (float): The reward obtained.
                - terminated (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was truncated.
                - info (Dict[str, Any]): Additional info.
        """
        position, current_status, *prize_indicators = self.state

        direction = 1 if action > 0 else -1
        next_pos = np.clip(position + direction, 0, self.length - 1)

        next_status = STATUS_TABLE.get(
            (self.previous_status, current_status), current_status
        )
        next_prize_indicators = self._compute_next_prize_indicators(
            next_pos, position, next_status, np.array(prize_indicators)
        )

        # Update state
        self.state[0] = next_pos
        self.state[1] = next_status
        self.state[2:] = next_prize_indicators

        reward = self._compute_reward(next_prize_indicators, action, position)
        self.previous_status = current_status

        return self._get_observation(), reward, False, False, {}

    def _compute_next_prize_indicators(
        self,
        next_position: float,
        position: float,
        next_status: int,
        prize_indicators: np.ndarray,
    ) -> np.ndarray:
        """Computes the next prize indicators based on the current state."""
        # next position is prize pos
        if int(next_position) == self.length - 1:
            # current pos is before prize pos (entering prize pos)
            if int(position) == self.length - 2:
                # if not max int, it will exit if and not enter elif or else, going straight to return 0
                if next_status == self.max_intensity:
                    return np.ones(self.num_prize_indicators, dtype=int)

            # if current and next positions are prize pos and indicators are 1
            elif all(prize_indicators == 1):
                return prize_indicators

            # if current and next positions are prize pos and indicators are not all 1
            else:
                return self._shift_prize_indicators(prize_indicators)
        # zero if next pos is not prize, or entering prize without max intensity
        return np.zeros(self.num_prize_indicators, dtype=int)

    def _shift_prize_indicators(self, prize_indicators: np.ndarray) -> np.ndarray:
        """Shifts the prize indicators to simulate prize movement."""
        if all(prize_indicators < 0.5):
            prize_indicators[0] = 1
            prize_indicators[1:] = np.zeros_like(prize_indicators[1:])
        else:
            one_index = np.argmax(prize_indicators)
            prize_indicators[one_index] = 0
            if one_index < self.num_prize_indicators - 1:
                prize_indicators[one_index + 1] = 1
        return prize_indicators

    def _compute_reward(
        self, next_prize_indicators: np.ndarray, action: int, position: float
    ) -> float:
        """Computes the reward based on the next prize indicators and action."""
        if all(next_prize_indicators == 1) and int(position) == self.length - 1:
            return 3.0
        return 0.0 if action == 0 else -1.0

    def _get_observation(self) -> np.ndarray:
        """Returns the current observation with optional noise."""
        if self.has_state_offset:
            obs = self.state.copy()
            obs[0] += self.position_offset
            obs[1] += self.status_indicator_offset
            obs[2:] += self.prize_indicator_offsets
            return obs
        return self.state.copy()

    def set_state(self, state: np.ndarray, previous_status: int) -> None:
        """Set environment state.

        Args:
            state (np.ndarray): _description_
            previous_status (int): _description_
        """
        self.state = state
        self.previous_status = previous_status
