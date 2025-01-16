"""Base class for planning agents."""

from typing import Dict, List, Tuple

import numpy as np

from bbi.agents import BaseQAgent
from bbi.environments import BaseEnv


class PlanningAgentBase(BaseQAgent):
    """Base class for planning agents that perform multi-step lookaheads using a dynamics model.

    This class:
    - Performs multi-step simulations to generate rewards and future Q-values.
    - Computes a list of TD targets.
    - Leaves weight computation (and potentially bounding box logic) to subclasses.
    """

    def __init__(
        self,
        number_actions: int = 2,
        number_positions: int = 11,
        number_intensities: int = 3,
        number_prize_indicators: int = 2,
        discount: float = 0.9,
        initial_value: float = 0.0,
    ):
        super().__init__(
            number_actions=number_actions,
            number_positions=number_positions,
            number_intensities=number_intensities,
            number_prize_indicators=number_prize_indicators,
            discount=discount,
            initial_value=initial_value,
        )

    def simulate_rollout(
        self,
        obs: Dict[str, float | int | np.ndarray],
        reward: float,
        max_horizon: int,
        done: bool,
        model: BaseEnv,
        **kwargs,
    ) -> Tuple[List[float], List[float]]:
        """Simulates a multi-step rollout using a learned or provided dynamics model.

        Args:
            obs (Observation): The initial state for the simulation.
            action (int): The initial action taken.
            reward (float): The immediate reward from the first step.
            next_obs (Observation): The next state after the initial action.
            max_horizon (int): The maximum number of lookahead steps.
            done (bool): Indicates if the episode has ended.

        Returns:
            Tuple[List[float], List[float]]: A list of rewards and a list of max future Q-values for each simulated step.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def compute_weights(self, td_targets: List[float], **kwargs) -> np.ndarray:
        """Calculates how each TD target should be weighted in the multi-step update.

        Args:
            td_targets (List[float]): The list of computed TD targets.

        Raises:
            NotImplementedError: Must be overridden to provide weighting logic.

        Returns:
            np.ndarray: An array of weights corresponding to each TD target.
        """
        raise NotImplementedError

    def compute_td_targets(
        self,
        rewards: List[float],
        max_future_values: List[float],
    ) -> List[float]:
        """Computes TD targets for each step of the multi-step simulation.

        Args:
            rewards (List[float]): A list of collected rewards over the rollout.
            max_future_values (List[float]): A list of max Q-values for the corresponding states.

        Returns:
            List[float]: The computed TD targets for each horizon step.
        """
        td_targets = []
        cumulative_reward = 0.0
        for h, reward_h in enumerate(rewards):
            cumulative_reward += (self.discount**h) * reward_h
            td_target = (
                cumulative_reward + (self.discount ** (h + 1)) * max_future_values[h]
            )
            td_targets.append(td_target)
        return td_targets

    def update_q_values(
        self,
        obs: Dict[str, float | int | np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, float | int | np.ndarray],
        alpha: float,
        max_horizon: int,
        model: BaseEnv,
        done: bool = False,
        **kwargs,
    ) -> float:
        """Implements a multi-step TD update using the planning logic.

        Args:
            state (np.ndarray): The current state observation.
            action (int): The action taken.
            reward (float): The immediate reward.
            next_state (np.ndarray): The next state observation.
            alpha (float): The learning rate for the Q-update.
            max_horizon (int): The maximum lookahead steps for planning.
            done (bool): Indicates if the episode has terminated.

        Returns:
            float: The TD error after applying the weighted multi-step update.
        """
        obs = self.round_obs(obs)
        next_obs = self.round_obs(next_obs)

        rewards, max_future_values = self.simulate_rollout(
            next_obs=next_obs,
            reward=reward,
            max_horizon=max_horizon,
            done=done,
            model=model,
        )

        td_targets = self.compute_td_targets(rewards, max_future_values)
        weights = self.compute_weights(td_targets, **kwargs)
        weighted_td_target = np.dot(weights, td_targets) # the division by the sum of weights is useless since the weights sum to 1

        td_error = (
            weighted_td_target
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
