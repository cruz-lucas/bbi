"""Base class for Q-based agents."""

from typing import Dict, List

import numpy as np

from bbi.utils.action_selection import compute_epsilon_greedy_action_probs


class BaseQAgent:
    """A base class for agents that maintain a Q-table."""

    def __init__(
        self,
        number_actions: int = 2,
        number_positions: int = 11,
        number_intensities: int = 3,
        number_prize_indicators: int = 2,
        discount: float = 0.9,
        initial_value: float = 0.0,
    ):
        """Initializes the base Q-agent with a Q-table for discrete states and actions.

        Args:
            explorer (BaseExplorer): Explorer that will choose the action.
            number_actions (int): The number of possible actions.
            number_positions (int): The size or length of the environment grid. Defaults to 11.
            number_intensities (int): Number of possible status intensities. Defaults to 3.
            number_prize_indicators (int): Number of prize indicator. Defaults to 2.
            discount (float): Discount factor for future rewards. Defaults to 0.9.
            initial_value (float): Initial Q-value for all state-action pairs. Defaults to 0.0.
        """
        self.number_actions = number_actions
        self.discount = discount
        self.number_positions = number_positions
        self.number_prize_indicators = number_prize_indicators

        self.q_values = np.full(
            (
                number_positions,
                number_intensities,
                2**number_prize_indicators,
                number_actions,
            ),
            fill_value=initial_value,
            dtype=float,
        )
        self.td_error: List[float] = []

    def round_obs(
        self, observation: Dict[str, float | int | np.ndarray]
    ) -> Dict[str, float | int | np.ndarray]:
        """Discretizes a continuous observation into a tuple of (position, intensity, prize).

        Args:
            observation (Dict[str, float | int | np.ndarray]): The raw (potentially continuous) observation.

        Returns:
            Dict[str, float | int | np.ndarray]: The discrete observation.
        """
        if observation["status_indicator"] <= 2.5:
            intensity_value = 0
        elif observation["status_indicator"] <= 7.5:
            intensity_value = 1
        else:
            intensity_value = 2

        if isinstance(observation["prize_indicators"], int):
            prize_indicators = observation["prize_indicators"]
        else:
            prize_indicators = sum(
                int(observation["prize_indicators"][i] >= 0.5) * (2**i)
                for i in range(self.number_prize_indicators)
            )

        return dict(
            {
                "position": int(np.round(observation["position"])),
                "status_indicator": int(intensity_value),
                "prize_indicators": prize_indicators,
            }
        )

    def get_action(
        self, observation: Dict[str, float | int | np.ndarray], epsilon: float
    ) -> int:
        """Chooses an action given the current state and a flag indicating greedy or exploratory mode.

        Args:
            observation (Dict[str, float | int | np.ndarray]): The current state observation.

        Returns:
            int: The chosen action index.
        """
        obs = self.round_obs(observation)

        q_values = self.q_values[
            obs["position"],
            obs["status_indicator"],
            obs["prize_indicators"],
        ]

        action_probs = compute_epsilon_greedy_action_probs(q_values, epsilon)
        return np.random.choice(len(action_probs), p=action_probs)

    def get_max_future_q(
        self, observation: Dict[str, float | int | np.ndarray], done: bool
    ) -> float:
        """Computes the maximum future Q-value for a given next state, respecting terminal conditions.

        Args:
            observation (Dict[str, float | int | np.ndarray]): The next state observation.
            done (bool): Indicates if the episode has terminated.

        Returns:
            float: The maximum Q-value over all actions in the next state.
        """
        if not done:
            max_future_q = np.max(
                self.q_values[
                    observation["position"],
                    observation["status_indicator"],
                    observation["prize_indicators"],
                ]
            )
        else:
            max_future_q = 0.0
        return max_future_q

    def get_min_future_q(
        self, observation: Dict[str, float | int | np.ndarray], done: bool
    ) -> float:
        """Computes the minimum future Q-value for a given next state, respecting terminal conditions.

        Args:
            observation (Dict[str, float | int | np.ndarray]): The next state observation.
            done (bool): Indicates if the episode has terminated.

        Returns:
            float: The minimum Q-value over all actions in the next state.
        """
        if not done:
            max_future_q = np.min(
                self.q_values[
                    observation["position"],
                    observation["status_indicator"],
                    observation["prize_indicators"],
                ]
            )
        else:
            max_future_q = 0.0
        return max_future_q

    def update_q_values(
        self,
        obs: Dict[str, float | int | np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, float | int | np.ndarray],
        alpha: float,
        **kwargs,
    ) -> float:
        """Updates the Q-table using a temporal-difference (TD) method. Must be implemented by subclasses.

        Args:
            observation (Observation): The current state observation.
            action (int): The action taken.
            reward (float): The reward received.
            next_obs (Observation): The resulting next state observation.
            alpha (float): The learning rate for the Q-update.
            **kwargs: Additional parameters for specialized Q-updates.

        Raises:
            NotImplementedError: If not overridden by a subclass.

        Returns:
            float: The TD error from this update.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
