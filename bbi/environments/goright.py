"""_summary_"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bbi.environments.base_env import BaseEnv

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


class GoRight(BaseEnv):
    """Custom Gymnasium environment for the "Go Right" task.

    The agent moves along a 1D grid, aiming to collect prizes at the end.
    """

    metadata = {"render_modes": ["human"], "environment_name": "GoRight"}

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = True,
        seed: Optional[int] = None,
        render_mode: Optional[str] = "human",
    ) -> None:
        """Initializes the GoRight environment.

        Args:
            num_prize_indicators: Number of prize indicators.
            env_length: Length of the grid.
            status_intensities: Possible status intensities.
            has_state_offset: Whether to add noise to observations.
            seed: Seed for reproducibility.
        """
        super().__init__()
        self.num_prize_indicators = num_prize_indicators
        self.length = env_length
        self.max_intensity = max(status_intensities)
        self.intensities = status_intensities
        self.previous_status: Optional[int] = None
        self.has_state_offset = has_state_offset

        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Box(
            low=0.0,
            high=max(env_length - 1, self.max_intensity, 1.0),
            shape=(2 + num_prize_indicators,),
            dtype=np.float32,
        )

        self.state: Optional[np.ndarray] = None
        self.render_mode = "human"
        self.last_action: Optional[int] = None
        self.last_pos = None
        self.last_reward = 0.0

        # Track cumulative reward and action count per episode
        self.total_reward = 0.0
        self.action_count = 0

        self.seed(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        """Sets the seed for the environment's random number generator.

        Args:
            seed: The seed value.
        """
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to its initial state.

        Args:
            seed: Seed for reproducibility.
            options: Additional options.

        Returns:
            observation: The initial observation of the environment.
            info: Additional info dictionary.
        """
        super().reset(seed=seed)
        self.state = np.zeros(2 + self.num_prize_indicators, dtype=np.float32)
        self.state[1] = self.np_random.choice(self.intensities)
        self.previous_status = self.np_random.choice(self.intensities)
        self.last_action = None
        self.last_pos = None
        self.last_reward = 0.0

        # Reset cumulative stats
        self.total_reward = 0.0
        self.action_count = 0

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
            action: The action taken by the agent (0: left, 1: right).

        Returns:
            observation: The next observation.
            reward: The reward obtained from taking this step.
            terminated: Whether the episode has ended.
            truncated: Whether the episode was truncated.
            info: Additional info dictionary.
        """
        if self.state is None:
            raise ValueError("State has not been initialized.")
        position, current_status, *prize_indicators = self.state

        next_pos = self._compute_next_position(action, position)

        next_status = STATUS_TABLE.get(
            (self.previous_status or 0, current_status or 0), current_status or 0
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
        self.last_action = action
        self.last_pos = position
        self.last_reward = reward

        # Update cumulative stats
        self.total_reward += reward
        self.action_count += 1

        return self._get_observation(), reward, False, False, {}

    def _compute_next_position(self, action: int, position: float):
        """Calculates the next position based on the current position and action.

        Args:
            action (int): The action taken by the agent (0: left, 1: right).
            position (float): _description_

        Returns:
            _type_: _description_
        """
        direction = 1 if action > 0 else -1
        return np.clip(position + direction, 0, self.length - 1)

    def _compute_next_prize_indicators(
        self,
        next_position: float,
        position: float,
        next_status: int,
        prize_indicators: np.ndarray,
    ) -> np.ndarray:
        """Computes the next prize indicators based on the current state.

        Args:
            next_position (float): The agent's next position in the grid.
            position (float): The agent's current position before moving.
            next_status (int): The next status intensity of the environment after the move.
            prize_indicators (np.ndarray): The current array of prize indicators before the move.

        Returns:
            np.ndarray: An updated array of prize indicators.
        """
        if int(next_position) == self.length - 1:
            if int(position) == self.length - 2:
                if next_status == self.max_intensity:
                    return np.ones(self.num_prize_indicators, dtype=int)
            elif all(prize_indicators == 1):
                return prize_indicators
            else:
                return self._shift_prize_indicators(prize_indicators)
        return np.zeros(self.num_prize_indicators, dtype=int)

    def _shift_prize_indicators(self, prize_indicators: np.ndarray) -> np.ndarray:
        """Shift prize indicators forward, simulating their movement.

        Args:
            prize_indicators (np.ndarray): Current prize indicators.

        Returns:
            np.ndarray: Updated prize indicators after shifting.
        """
        if all(prize_indicators < 0.5):
            prize_indicators[0] = 1
            prize_indicators[1:] = 0
        else:
            one_index = np.argmax(prize_indicators)
            prize_indicators[one_index] = 0
            if one_index < self.num_prize_indicators - 1:
                prize_indicators[one_index + 1] = 1
        return prize_indicators

    def _compute_reward(
        self, next_prize_indicators: np.ndarray, action: int, position: float
    ) -> float:
        """Compute the reward based on action and prize indicators.

        Args:
            next_prize_indicators (np.ndarray): Updated prize indicators.
            action (int): Action taken by the agent.
            position (float): Agent's current position.

        Returns:
            float: Calculated reward.
        """
        if all(next_prize_indicators == 1) and int(position) == self.length - 1:
            return 3.0
        return 0.0 if action == self.LEFT else -1.0

    def _get_observation(self) -> np.ndarray:
        """Get the current observation with optional offsets.

        Returns:
            np.ndarray: The current observation.
        """
        if self.state is None:
            raise ValueError("State has not been initialized.")

        if self.has_state_offset:
            obs = self.state.copy()
            obs[0] += self.position_offset
            obs[1] += self.status_indicator_offset
            obs[2:] += self.prize_indicator_offsets
            return obs
        return self.state.copy()

    def set_state(self, state: np.ndarray, previous_status: int) -> None:
        """Set the environment state.

        Args:
            state (np.ndarray): The new state.
            previous_status (int): Previous status intensity.
        """
        self.state = state
        self.previous_status = previous_status

    def set_state_from_rounded(self, state: np.ndarray, previous_status: int) -> None:
        """Set the environment state.

        Args:
            state (np.ndarray): The new state.
            previous_status (int): Previous status intensity.
        """
        prize = [int(i) for i in list(bin(state[-1])[2:])]
        if len(prize) == 1:
            prize = [0] + prize
        self.state = np.array([state[0], state[1] * 5] + prize)
        self.previous_status = previous_status * 5
