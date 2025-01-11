"""Unselective Planning Agent"""

from typing import Dict, List, Tuple

import numpy as np

from bbi.agents import PlanningAgentBase
from bbi.environments import BaseEnv


class UnselectivePlanningAgent(PlanningAgentBase):
    """An agent that performs multi-step TD updates using a dynamics model, with equal (unselective) weighting.

    Args:
        PlanningAgentBase (_type_): _description_
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
        next_obs: Dict[str, float | int | np.ndarray],
        reward: float,
        max_horizon: int,
        done: bool,
        model: BaseEnv,
    ) -> Tuple[List[float], List[float]]:
        """Simulates a multi-step rollout using a learned or provided dynamics model.

        Args:
            state (np.ndarray): The initial state for the simulation.
            action (int): The initial action taken.
            reward (float): The immediate reward from the first step.
            next_state (np.ndarray): The next state after the initial action.
            max_horizon (int): The maximum number of lookahead steps.
            done (bool): Indicates if the episode has ended.

        Returns:
            Tuple[List[float], List[float]]: A list of rewards and a list of max future Q-values for each simulated step.
        """
        rewards = [reward]
        terminated = done
        truncated = False

        max_future_values = [self.get_max_future_q(next_obs, terminated or truncated)]

        simulated_observation = next_obs

        for h in range(1, max_horizon + 1):
            if terminated or truncated:
                break
            action_h = self.get_action(simulated_observation, epsilon=0.0)  # greedy
            next_simulated_obs, reward_h, terminated, truncated, _ = model.step(
                action_h
            )

            next_obs_rounded = self.round_obs(next_simulated_obs)

            rewards.append(reward_h)
            max_future_values.append(
                self.get_max_future_q(next_obs_rounded, terminated or truncated)
            )
            simulated_observation = next_simulated_obs

        return rewards, max_future_values

    def compute_weights(self, td_targets: List[float], **kwargs) -> np.ndarray:
        """Assigns equal weights to each horizon step in the multi-step TD update.

        Args:
            td_targets (List[float]): Computed TD targets for each step.

        Returns:
            np.ndarray: An array of equal weights, summing to 1.
        """
        return np.ones(len(td_targets)) / len(td_targets)
