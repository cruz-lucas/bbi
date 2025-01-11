"""Q-Learning Agent Implementation"""

from typing import Dict

import numpy as np

from bbi.agents import BaseQAgent


class QLearningAgent(BaseQAgent):
    """A Q-Learning agent that uses a tabular 1-step update (no model, no horizon, no tau)."""

    def update_q_values(
        self,
        obs: Dict[str, float | int | np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, float | int | np.ndarray],
        alpha: float,
        done: bool = False,
        **kwargs,
    ) -> float:
        """Initializes a selective planning agent with multi-step lookahead and bounding-box logic.

        Args:
            action_space (gymnasium.Space): Action space for the environment.
            gamma (float): Discount factor.
            environment_length (int): Size/length of the environment grid.
            intensities (np.ndarray | List[int]): Possible status intensities.
            num_prize_indicators (int): Number of prize indicator bits.
            initial_value (float): Initial Q-value for all state-action pairs.
            tau (float): Temperature parameter for weighting logic.
            model_id (str): Specifies the type of dynamics model to use.
            learning_rate (float): Learning rate for any learnable dynamics model.
        """
        obs = self.round_obs(obs)
        next_obs = self.round_obs(next_obs)

        # 1-step TD update:
        max_future_q = self.get_max_future_q(next_obs, done)
        td_target = reward + self.discount * max_future_q * (0.0 if done else 1.0)
        td_error = (
            td_target
            - self.q_values[
                obs["position"],
                obs["status_indicator"],
                obs["prize_indicators"],
                action,
            ]
        )
        self.q_values[
            obs["position"],
            obs["status_indicator"],
            obs["prize_indicators"],
            action,
        ] += alpha * td_error
        self.td_error.append(td_error)

        return td_error
