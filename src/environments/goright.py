"""Module for the GoRightEnv Gymnasium environment."""

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
    (10, 10): 0,
}


class GoRightEnv(gym.Env):
    """Custom Gymnasium environment for the "Go Right" task.

    This environment simulates an agent moving along a one-dimensional grid of specified length. The
    agent can move left or right, aiming to reach the end of the grid to collect prizes. The environment
    includes status indicators and intensities that influence the agent's reward and state transitions.
    The state can include optional random offsets to simulate observation noise.

    Attributes:
        num_prize_indicators (int): Number of prize indicators in the environment.
        length (int): Length of the one-dimensional grid.
        max_intensity (int): Maximum value of status intensities.
        intensities (List[int]): List of possible status intensities.
        previous_status (Optional[int]): The previous status intensity.
        has_state_offset (bool): Whether to add random offsets to the state observations.
        action_space (gym.spaces.Discrete): Action space of the environment.
        observation_space (gym.spaces.Dict): Observation space of the environment.
    """

    def __init__(
        self,
        num_prize_indicators: int = 2,
        length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the GoRight environment.

        Args:
            num_prize_indicators (int): Number of prize indicators in the environment. Defaults to 2.
            length (int): Length of the one-dimensional grid. Defaults to 11.
            status_intensities (List[int]): List of possible status intensities. Defaults to [0, 5, 10].
            has_state_offset (bool): Whether to add random offsets to the state observations. Defaults to False.
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        """
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
                "prize_status": spaces.Box(
                    low=np.zeros(num_prize_indicators),
                    high=np.ones(num_prize_indicators),
                    shape=(num_prize_indicators,),
                    dtype=np.int8,
                ),
            }
        )

        self.reset(seed=seed)

    def seed(self, seed=None):
        """Sets the seed for the environment's random number generator.

        Args:
            seed (Optional[int]): The seed value for the random number generator. If None, a random seed is used.

        Returns:
            List[int]: A list containing the seed.
        """
        self.np_random, seed = gym.utils.seeding.np_random(
            int(np.random.random(1)[0] * 100 if seed is None else seed)
        )
        return [seed]

    def reset(
        self,
        seed: Optional[int] = None,
        new_state: Optional[np.ndarray] = None,
        previous_status: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to its initial state.

        Args:
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
            new_state (Optional[np.ndarray]): An optional state to reset to. If None, initializes to default starting state.
            previous_status (Optional[int]): An optional previous status intensity. If None, randomly selected.

        Returns:
            Tuple[np.ndarray, Dict]: A tuple containing the initial observation and an empty info dictionary.
        """
        self.seed(seed)

        if new_state is not None:
            self.state: np.ndarray = new_state
            self.previous_status: int = previous_status
        else:
            self.state: np.ndarray = np.zeros(
                2 + self.num_prize_indicators, dtype=float
            )
            self.state[1] = self.np_random.choice(self.intensities)
            self.previous_status: int = self.np_random.choice(self.intensities)

        if self.has_state_offset:
            self.position_offset = self.np_random.uniform(-0.25, 0.25)
            self.status_indicator_offset = self.np_random.uniform(-1.25, 1.25)
            self.prize_indicator_offsets = self.np_random.uniform(
                -0.25, 0.25, size=self.num_prize_indicators
            )

        return self._add_offset_to_state(self.state.copy()), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment given an action.

        Args:
            action (int): The action to be taken by the agent. 0 for left, 1 for right.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: A tuple containing:
                - observation (np.ndarray): The next observation of the environment.
                - reward (float): The reward obtained from taking the action.
                - done (bool): Whether the episode has ended (always False in this environment).
                - truncated (bool): Whether the episode was truncated (always False in this environment).
                - info (Dict[str, Any]): Additional information about the step.
        """
        position, current_status, *prize_indicators = self.state

        direction = 1 if action > 0 else -1
        next_pos = np.clip(
            position + direction, a_min=0, a_max=self.length - 1
        ) 

        next_status = STATUS_TABLE.get((self.previous_status, current_status), None)
        next_prize_indicators = self._compute_next_prize_indicators(
            next_pos, position, next_status, prize_indicators
        )

        # Update the state
        self.state[0] = next_pos
        self.state[1] = next_status
        self.state[2:] = next_prize_indicators

        reward = self._compute_reward(next_prize_indicators, action)
        self.previous_status = current_status

        return self._add_offset_to_state(self.state.copy()), reward, False, False, {}

    def _compute_next_prize_indicators(
        self, next_position: int, position: int, next_status: int, prize_indicators: np.ndarray
    ) -> np.ndarray:
        """Computes the next prize indicators based on the current state.

        Args:
            next_position (int): The agent's next position after taking the action.
            position (int): The agent's current position.
            next_status (int): The next status intensity.
            prize_indicators (np.ndarray): The current prize indicators.

        Returns:
            np.ndarray: The updated prize indicators.
        """
        if (
            next_position == self.length - 1
        ):  # If the agent is moving to the last position
            if (
                position == self.length - 2
            ):  # If the agent is currently at the second-to-last position
                if next_status == self.max_intensity:  # If the intensity is at maximum
                    return np.ones(self.num_prize_indicators, dtype=int)
            else:  # Agent is already at the last position
                if all(prize_indicators) == 1:  # Receiving prize
                    return prize_indicators
                else:  # Not receiving prize
                    return self._shift_prize_indicators(prize_indicators)

        return np.zeros(self.num_prize_indicators, dtype=int)

    def _shift_prize_indicators(self, prize_indicators: np.ndarray) -> np.ndarray:
        """Shifts the prize indicators to simulate the prize movement.

        Args:
            prize_indicators (np.ndarray): The current prize indicators.

        Returns:
            np.ndarray: The updated prize indicators after shifting.
        """
        if np.sum(prize_indicators) == 0:  # All zeros
            prize_indicators[0] = 1
        else:  # Has a 1 in it, shift it
            one_index = np.argmax(prize_indicators)
            if (
                one_index == self.num_prize_indicators - 1
            ):  # If one in the last place, return all zeros
                prize_indicators[one_index] = 0
            else:
                prize_indicators[one_index] = 0
                prize_indicators[one_index + 1] = 1

        return prize_indicators

    def _compute_reward(self, next_prize_indicators: np.ndarray, action: int) -> int:
        """Computes the reward based on the next prize indicators and action.

        Args:
            next_prize_indicators (np.ndarray): The prize indicators after the action.
            action (int): The action taken by the agent.

        Returns:
            int: The reward for the given action.
        """
        if all(next_prize_indicators) == 1:
            return 3
        if action == 0:  # Left
            return 0
        return -1  # Right

    def _add_offset_to_state(self, state: np.ndarray) -> np.ndarray:
        """Adds random noise to the state observations if state offsets are enabled.

        Args:
            state (np.ndarray): The original state without noise.

        Returns:
            np.ndarray: The state with added noise.
        """
        if self.has_state_offset:
            state[0] = state[0] + self.position_offset
            state[1] = state[1] + self.status_indicator_offset
            state[2:] = state[2:] + self.prize_indicator_offsets
        return state
