"""Selective Planning Agent"""

from typing import List, Tuple, Dict

import gymnasium
import numpy as np

from bbi.agents import PlanningAgentBase
from bbi.models import BBI


class SelectivePlanningAgent(PlanningAgentBase):
    """An agent that performs multi-step TD updates with a dynamics model and selective weighting using tau.

    Args:
        PlanningAgentBase (_type_): _description_
    """

    def __init__(
        self,
        action_space: gymnasium.Space,
        gamma: float = 0.9,
        environment_length: int = 11,
        intensities: np.ndarray | List[int] = [0, 5, 10],
        num_prize_indicators: int = 2,
        initial_value: float = 0.0,
        tau: float = 1.0,
    ):
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
        super().__init__(
            action_space=action_space,
            gamma=gamma,
            environment_length=environment_length,
            intensities=intensities,
            num_prize_indicators=num_prize_indicators,
            initial_value=initial_value,
        )
        self.tau = tau

    def simulate_rollout(
        self,
        next_obs: Dict[str, float | int | np.ndarray],
        reward: float,
        max_horizon: int,
        done: bool,
        model: BBI,
    ):
        """Extends the base rollout to track upper and lower reward/Q-value bounds for bounding-box inference.

        Args:
            state (np.ndarray): The initial state before planning.
            action (int): The initial action to simulate.
            reward (float): Immediate reward from the real environment step.
            next_state (np.ndarray): Next state after the first step.
            max_horizon (int): Number of planning steps to simulate.
            done (bool): Whether this transition was terminal.

        Returns:
            Tuple[List[float], List[float]]: The standard rollout's reward list and max future values.
        """
        rewards = [reward]
        terminated = done
        truncated = False

        max_future_values = [self.get_max_future_q(next_obs, terminated or truncated)]

        simulated_observation = next_obs

        upper_rewards = [reward]
        lower_rewards = [reward]

        upper_max_future_values = max_future_values.copy()
        lower_max_future_values = max_future_values.copy()

        for _ in range(1, max_horizon):
            if terminated or truncated:
                break

            action_h = self.get_action(simulated_observation, greedy=True)
            next_simulated_obs, reward_h, terminated, truncated, _ = (
                model.step(action_h)
            )

            done = terminated or truncated

            rewards.append(reward_h)
            max_future_values.append(self.get_max_future_q(next_simulated_obs, done))

            # Bounding box rewards
            # model.get_next_bounds(simulated_observation)
            upper_rewards.append(np.max(model.reward_bounding_box))
            lower_rewards.append(np.min(model.reward_bounding_box))

            # define action bound to select action for bounds
            upper_q_value_bounds = [
                self.get_max_future_q(state_bound, done)
                for state_bound in model.state_bounding_box
            ]
            lower_q_value_bounds = [
                self.get_min_future_q(state_bound, done)
                for state_bound in model.state_bounding_box
            ]

            upper_max_future_values.append(np.max(upper_q_value_bounds))
            lower_max_future_values.append(np.min(lower_q_value_bounds))

            simulated_observation = next_simulated_obs.copy()

        # Store bounding box data for later weighting
        self._upper_rewards = upper_rewards
        self._lower_rewards = lower_rewards
        self._upper_max_future_values = upper_max_future_values
        self._lower_max_future_values = lower_max_future_values

        return rewards, max_future_values

    def compute_weights(self, td_targets: List[float], **kwargs) -> np.ndarray:
        """Implements the selective weighting mechanism for bounding-box TD targets.

        Args:
            td_targets (List[float]): The nominal TD targets for each horizon step.

        Returns:
            np.ndarray: An array of weights computed using bounding-box uncertainty and tau.
        """
        lower_td_targets, upper_td_targets = self._compute_bounding_box_td_targets(
            td_targets
        )
        # Negative uncertainties = lower - upper
        neg_uncertainties = lower_td_targets - upper_td_targets
        weights = self._calculate_weigths(
            neg_uncertainties=neg_uncertainties, tau=self.tau
        )

        assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
        return weights


    def _compute_bounding_box_td_targets(
        self, td_targets: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates upper and lower TD targets using bounding-box estimates of rewards and Q-values.

        Args:
            td_targets (List[float]): The base TD targets ignoring uncertainty.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of lower and upper TD targets per horizon step.
        """
        lower_td_targets = []
        upper_td_targets = []
        lower_cumulative_reward = 0.0
        upper_cumulative_reward = 0.0

        for h in range(len(td_targets)):
            # Upper
            upper_cumulative_reward += (self.discount**h) * self._upper_rewards[h]
            upper_td_target = (
                upper_cumulative_reward
                + (self.discount ** (h + 1)) * self._upper_max_future_values[h]
            )
            upper_td_targets.append(upper_td_target)

            # Lower
            lower_cumulative_reward += (self.discount**h) * self._lower_rewards[h]
            lower_td_target = (
                lower_cumulative_reward
                + (self.discount ** (h + 1)) * self._lower_max_future_values[h]
            )
            lower_td_targets.append(lower_td_target)

        return np.array(lower_td_targets), np.array(upper_td_targets)

    def _calculate_weigths(self, neg_uncertainties: np.ndarray, tau: float):
        """Converts negative uncertainties into exponential weights for TD targets.

        Args:
            neg_uncertainties (np.ndarray): Negative uncertainties from bounding-box analysis.
            tau (float): Temperature parameter controlling weighting sensitivity.

        Returns:
            np.ndarray: Normalized weights for each time step.
        """
        scaled_targets = neg_uncertainties / tau
        exp_values = np.exp(scaled_targets)
        return exp_values / np.sum(exp_values)
