"""Unselective Planning Agent"""

from typing import List

import gymnasium
import numpy as np

from bbi.agents import BaseQAgent
from bbi.environments import GoRight
from bbi.models import ExpectationModel, SamplingModel


class UnselectivePlanningAgent(BaseQAgent):
    """An agent that performs multi-step TD updates using a dynamics model, but no tau selection.

    This agent computes multi-step returns by simulating forward up to max_horizon steps
    and then averaging the TD targets (unselective = equal weights).
    """

    def __init__(
        self,
        action_space: gymnasium.Space,
        gamma: float = 0.9,
        environment_length: int = 11,
        intensities: np.ndarray | List[int] = [0, 5, 10],
        num_prize_indicators: int = 2,
        initial_value: float = 0.0,
        debug: bool = False,
        model_type: str = "Perfect",
    ):
        super().__init__(
            action_space,
            gamma,
            environment_length,
            intensities,
            num_prize_indicators,
            initial_value,
            debug,
        )
        if model_type == "perfect":
            self.dynamics_model = GoRight(
                num_prize_indicators=num_prize_indicators,
                env_length=environment_length,
                has_state_offset=False,
            )
        elif model_type == "expected":
            self.dynamics_model = ExpectationModel(
                num_prize_indicators=num_prize_indicators,
                env_length=environment_length,
                has_state_offset=False,
            )
        elif model_type == "sampling":
            self.dynamics_model = SamplingModel(
                num_prize_indicators=num_prize_indicators,
                env_length=environment_length,
                has_state_offset=False,
            )
        else:
            raise ValueError(
                'Please specify the type of model for unselective planning: "Perfect", "Expectation", or "Sampling".'
            )

    def update_q_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        alpha: float,
        max_horizon: int,
        previous_status: int,  # only used if using perfect model
        done: bool = False,
        **kwargs,
    ) -> float:
        pos, intensity, prize = self.round_obs(state)

        # Simulate forward for up to max_horizon steps
        rewards = [reward]
        simulated_state = next_state.copy()
        terminated = done
        truncated = False

        max_future_values = [
            self.get_max_future_q(simulated_state, terminated or truncated)
        ]
        self.dynamics_model.set_state(state=simulated_state, previous_status=state[1])

        for _ in range(1, max_horizon):
            if terminated or truncated:
                break
            action_h = self.get_action(simulated_state, greedy=True)
            next_simulated_state, reward_h, terminated, truncated, _ = (
                self.dynamics_model.step(action_h)
            )
            rewards.append(reward_h)
            simulated_state = next_simulated_state.copy()
            max_future_values.append(
                self.get_max_future_q(simulated_state, terminated or truncated)
            )

        # Compute TD targets
        td_targets = []
        cumulative_reward = 0.0
        for h, reward_h in enumerate(rewards):
            cumulative_reward += (self.gamma**h) * reward_h
            td_target = (
                cumulative_reward + (self.gamma ** (h + 1)) * max_future_values[h]
            )
            td_targets.append(td_target)

        # Unselective = equal weighting
        weighted_td_target = np.mean(td_targets)
        td_error = weighted_td_target - self.q_values[pos, intensity, prize, action]
        self.q_values[pos, intensity, prize, action] += alpha * td_error
        self.td_error.append(td_error)

        if self.debug:
            print(
                {
                    "function": "update_q_values",
                    "state": state,
                    "action": action,
                    "td_targets": td_targets,
                    "weighted_td_target": weighted_td_target,
                    "td_error": td_error,
                }
            )

        return td_error
