"""Selective Planning Agent"""

from typing import List, Tuple

import gymnasium
import numpy as np

from bbi.agents import PlanningAgentBase
from bbi.models import BBI


class SelectivePlanningAgent(PlanningAgentBase):
    """An agent that performs multi-step TD updates with a dynamics model and selective weighting using tau."""

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
        super().__init__(
            action_space=action_space,
            gamma=gamma,
            environment_length=environment_length,
            intensities=intensities,
            num_prize_indicators=num_prize_indicators,
            initial_value=initial_value,
        )
        self.dynamics_model = BBI(
            num_prize_indicators=num_prize_indicators,
            env_length=environment_length,
            has_state_offset=False,
        )
        self.tau = tau

    def simulate_rollout(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        max_horizon: int,
        done: bool,
    ):
        """
        Extend the base rollout to also track bounding box rewards and values.
        """
        rewards = [reward]
        simulated_state = next_state.copy()
        terminated = done
        truncated = False

        max_q = self.get_max_future_q(simulated_state, terminated or truncated)
        max_future_values = [max_q]

        self.set_model_state(state=simulated_state, previous_state=state)

        upper_rewards = [reward]
        lower_rewards = [reward]

        upper_max_future_values = [max_q]
        lower_max_future_values = [max_q]

        for _ in range(1, max_horizon):
            if terminated or truncated:
                break

            action_h = self.get_action(simulated_state, greedy=True)
            next_simulated_state, reward_h, terminated, truncated, _ = (
                self.dynamics_model.step(action_h)
            )

            done = terminated or truncated

            rewards.append(reward_h)
            max_future_values.append(self.get_max_future_q(next_simulated_state, done))

            # Bounding box rewards
            self.dynamics_model.get_next_bounds(simulated_state)
            upper_rewards.append(np.max(self.dynamics_model.reward_bounding_box))
            lower_rewards.append(np.min(self.dynamics_model.reward_bounding_box))

            # define action bound to select action for bounds
            upper_q_value_bounds = [
                self.get_max_future_q(state_bound, done)
                for state_bound in self.dynamics_model.state_bounding_box
            ]
            lower_q_value_bounds = [
                self.get_min_future_q(state_bound, done)
                for state_bound in self.dynamics_model.state_bounding_box
            ]

            upper_max_future_values.append(np.max(upper_q_value_bounds))
            lower_max_future_values.append(np.min(lower_q_value_bounds))

            simulated_state = next_simulated_state.copy()

        # Store bounding box data for later weighting
        self._upper_rewards = upper_rewards
        self._lower_rewards = lower_rewards
        self._upper_max_future_values = upper_max_future_values
        self._lower_max_future_values = lower_max_future_values

        return rewards, max_future_values

    def compute_weights(self, td_targets: List[float], **kwargs) -> np.ndarray:
        lower_td_targets, upper_td_targets = self._compute_bounding_box_td_targets(
            td_targets
        )
        # Negative uncertainties = lower - upper
        neg_uncertainties = lower_td_targets - upper_td_targets
        return self._calculate_weigths(
            neg_uncertainties=neg_uncertainties, tau=self.tau
        )

    def _compute_bounding_box_td_targets(
        self, td_targets: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        lower_td_targets = []
        upper_td_targets = []
        lower_cumulative_reward = 0.0
        upper_cumulative_reward = 0.0

        for h in range(len(td_targets)):
            # Upper
            upper_cumulative_reward += (self.gamma**h) * self._upper_rewards[h]
            upper_td_target = (
                upper_cumulative_reward
                + (self.gamma ** (h + 1)) * self._upper_max_future_values[h]
            )
            upper_td_targets.append(upper_td_target)

            # Lower
            lower_cumulative_reward += (self.gamma**h) * self._lower_rewards[h]
            lower_td_target = (
                lower_cumulative_reward
                + (self.gamma ** (h + 1)) * self._lower_max_future_values[h]
            )
            lower_td_targets.append(lower_td_target)

        return np.array(lower_td_targets), np.array(upper_td_targets)

    def _calculate_weigths(self, neg_uncertainties: np.ndarray, tau: float):
        scaled_targets = neg_uncertainties / tau
        exp_values = np.exp(scaled_targets)
        return exp_values / np.sum(exp_values)
