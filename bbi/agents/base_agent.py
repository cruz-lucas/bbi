"""Base class for Q-based agents."""

from typing import List, Tuple

import gymnasium
import numpy as np


class BaseQAgent:
    """A base class for agents that maintain a Q-table."""

    def __init__(
        self,
        action_space: gymnasium.Space,
        gamma: float = 0.9,
        environment_length: int = 11,
        intensities: np.ndarray | List[int] = [0, 5, 10],
        num_prize_indicators: int = 2,
        initial_value: float = 0.0,
    ):
        """Initializes the base Q-agent with a Q-table for discrete states and actions.

        Args:
            action_space (gymnasium.Space): The space of possible actions.
            gamma (float): Discount factor for future rewards. Defaults to 0.9.
            environment_length (int): The size or length of the environment grid. Defaults to 11.
            intensities (np.ndarray | List[int]): Possible status intensities. Defaults to [0, 5, 10].
            num_prize_indicators (int): Number of prize indicator bits. Defaults to 2.
            initial_value (float): Initial Q-value for all state-action pairs. Defaults to 0.0.
        """
        self.action_space = action_space
        self.gamma = gamma
        self.environment_length = environment_length
        self.num_prize_indicators = num_prize_indicators

        self.q_values = np.full(
            (
                environment_length,
                len(intensities),
                2**num_prize_indicators,
                action_space.n,
            ),
            fill_value=initial_value,
            dtype=float,
        )
        self.td_error: List[float] = []

    def round_obs(self, obs: np.ndarray) -> Tuple[int, int, int]:
        """Discretizes a continuous observation into a tuple of (position, intensity, prize).

        Args:
            obs (np.ndarray): The raw (potentially continuous) observation.

        Returns:
            Tuple[int, int, int]: The discrete state indices (position, intensity, prize).
        """
        if obs[1] <= 2.5:
            intensity_value = 0
        elif obs[1] <= 7.5:
            intensity_value = 1
        else:
            intensity_value = 2

        prize_indicator = sum(
            int(obs[2 + i] >= 0.5) * (2**i) for i in range(self.num_prize_indicators)
        )

        return (
            int(round(obs[0])),
            intensity_value,
            prize_indicator,
        )

    def get_action(self, state: np.ndarray, greedy: bool) -> int:
        """Chooses an action given the current state and a flag indicating greedy or exploratory mode.

        Args:
            state (np.ndarray): The current state observation.
            greedy (bool): If True, selects an action using the highest Q-value; otherwise random.

        Returns:
            int: The chosen action index.
        """
        pos, intensity, prize = self.round_obs(state)

        if not greedy:
            return self.action_space.sample()

        q_values = self.q_values[pos, intensity, prize]
        max_value = np.max(q_values)
        ties = np.flatnonzero(q_values == max_value)

        return int(np.random.choice(ties))

    def get_max_future_q(self, state: np.ndarray, done: bool) -> float:
        """Computes the maximum future Q-value for a given next state, respecting terminal conditions.

        Args:
            state (np.ndarray): The next state observation.
            done (bool): Indicates if the episode has terminated.

        Returns:
            float: The maximum Q-value over all actions in the next state.
        """
        if not done:
            pos, intensity, prize = self.round_obs(state)
            max_future_q = np.max(self.q_values[pos, intensity, prize])
        else:
            max_future_q = 0.0
        return max_future_q

    def get_min_future_q(self, state: np.ndarray, done: bool) -> float:
        """Computes the minimum future Q-value for a given next state, respecting terminal conditions.

        Args:
            state (np.ndarray): The next state observation.
            done (bool): Indicates if the episode has terminated.

        Returns:
            float: The minimum Q-value over all actions in the next state.
        """
        if not done:
            pos, intensity, prize = self.round_obs(state)
            max_future_q = np.min(self.q_values[pos, intensity, prize])
        else:
            max_future_q = 0.0
        return max_future_q

    def update_q_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        alpha: float,
        **kwargs,
    ) -> float:
        """Updates the Q-table using a temporal-difference (TD) method. Must be implemented by subclasses.

        Args:
            state (np.ndarray): The current state observation.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The resulting next state.
            alpha (float): The learning rate for the Q-update.
            **kwargs: Additional parameters for specialized Q-updates.

        Raises:
            NotImplementedError: If not overridden by a subclass.

        Returns:
            float: The TD error from this update.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
