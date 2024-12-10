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
        debug: bool = False,
    ):
        self.action_space = action_space
        self.gamma = gamma
        self.environment_length = environment_length
        self.num_prize_indicators = num_prize_indicators
        self.debug = debug

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
        """Discretizes the continuous observation into discrete state components."""
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
        """Selects an action based on the current state."""
        pos, intensity, prize = self.round_obs(state)

        if not greedy:
            return self.action_space.sample()

        q_values = self.q_values[pos, intensity, prize]
        max_value = np.max(q_values)
        ties = np.flatnonzero(q_values == max_value)

        if self.debug:
            print(
                {
                    "function": "get_action",
                    "state": state,
                    "q_values": q_values,
                    "ties": ties,
                }
            )

        return int(np.random.choice(ties))

    def get_max_future_q(self, state: np.ndarray, done: bool) -> float:
        """Retrieves the maximum future Q-value for a given state."""
        if not done:
            pos, intensity, prize = self.round_obs(state)
            max_future_q = np.max(self.q_values[pos, intensity, prize])
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
        """Abstract method: Implement in subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")
