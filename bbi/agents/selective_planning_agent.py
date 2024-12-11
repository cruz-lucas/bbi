"""Selective Planning Agent"""

from typing import List

import gymnasium
import numpy as np

from bbi.agents import BaseQAgent
from bbi.models import BBI


class SelectivePlanningAgent(BaseQAgent):
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
            action_space,
            gamma,
            environment_length,
            intensities,
            num_prize_indicators,
            initial_value,
        )
        self.dynamics_model = BBI(
            num_prize_indicators=num_prize_indicators,
            env_length=environment_length,
            has_state_offset=False,
        )
        self.tau = tau

    def update_q_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        alpha: float,
        max_horizon: int,
        done: bool = False,
        **kwargs,
    ) -> float:
        pos, intensity, prize = self.round_obs(state)

        # Simulate forward up to max_horizon steps
        rewards = [reward]
        simulated_state = next_state.copy()
        terminated = done
        truncated = False

        max_q = self.get_max_future_q(simulated_state, terminated or truncated)
        max_future_values = [max_q]
        self.dynamics_model.set_state(state=simulated_state, previous_status=state[1])

        upper_rewards = [reward]
        lower_rewards = [reward]

        upper_max_future_values = [max_q]
        lower_max_future_values = [max_q]

        for _ in range(1, max_horizon):
            if terminated or truncated:
                break

            # Expectation model rollout
            action_h = self.get_action(simulated_state, greedy=True)
            next_simulated_state, reward_h, terminated, truncated, _ = (
                self.dynamics_model.step(action_h)
            )
            rewards.append(reward_h)
            simulated_state = next_simulated_state.copy()
            max_future_values.append(
                self.get_max_future_q(simulated_state, terminated or truncated)
            )

            # BBI rollout
            # Upper
            upper_rewards.append(np.max(self.dynamics_model.reward_bounding_box))
            upper_max_future_values.append(
                self.get_max_future_q(
                    self.dynamics_model.state_bounding_box[1], terminated or truncated
                )
            )

            # Lower
            lower_rewards.append(np.min(self.dynamics_model.reward_bounding_box))
            lower_max_future_values.append(
                self.get_max_future_q(
                    self.dynamics_model.state_bounding_box[0], terminated or truncated
                )
            )

        # print("rewards: ", rewards)
        # print("upper_rewards: ", upper_rewards)
        # print("lower_rewards: ", lower_rewards)
        # print("upper_max_future_values: ", upper_max_future_values)
        # print("lower_max_future_values: ", lower_max_future_values)

        # Compute TD targets
        td_targets = []
        cumulative_reward = 0.0

        lower_td_targets = []
        upper_td_targets = []
        lower_cumulative_reward = 0.0
        upper_cumulative_reward = 0.0

        for h, reward_h in enumerate(rewards):
            # Expectation TD Targets
            cumulative_reward += (self.gamma**h) * reward_h
            td_target = (
                cumulative_reward + (self.gamma ** (h + 1)) * max_future_values[h]
            )
            td_targets.append(td_target)

            # Bounding box - TD targets
            # Upper
            upper_cumulative_reward += (self.gamma**h) * upper_rewards[h]

            upper_td_target = (
                upper_cumulative_reward
                + (self.gamma ** (h + 1)) * upper_max_future_values[h]
            )
            upper_td_targets.append(upper_td_target)

            # Lower
            lower_cumulative_reward += (self.gamma**h) * lower_rewards[h]
            lower_td_target = (
                lower_cumulative_reward
                + (self.gamma ** (h + 1)) * lower_max_future_values[h]
            )
            lower_td_targets.append(lower_td_target)

            # print("h: ", h)
            # print("lower_cumulative_reward: ", lower_cumulative_reward)
            # print("upper_cumulative_reward: ", upper_cumulative_reward)
            # print("lower_td_target: ", lower_td_target)
            # print("upper_td_target: ", upper_td_target)
            # print("lower_td_targets: ", lower_td_targets)
            # print("upper_td_targets: ", upper_td_targets)

        upper_td_targets = np.array(upper_td_targets)
        lower_td_targets = np.array(lower_td_targets)

        # print("lower: ", lower_td_targets)
        # print("upper: ", upper_td_targets)
        # print("diff: ", lower_td_targets - upper_td_targets)

        weights = self._calculate_weigths(
            neg_uncertainties=lower_td_targets - upper_td_targets, tau=self.tau
        )

        weighted_td_target = np.dot(weights, td_targets)
        td_error = weighted_td_target - self.q_values[pos, intensity, prize, action]
        self.q_values[pos, intensity, prize, action] += alpha * td_error
        self.td_error.append(td_error)

        return td_error

    def _calculate_weigths(self, neg_uncertainties: np.array, tau: float):
        scaled_targets = neg_uncertainties / tau
        exp_values = np.exp(scaled_targets)
        return exp_values / np.sum(exp_values)
