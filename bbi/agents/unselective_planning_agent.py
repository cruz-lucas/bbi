"""Unselective Planning Agent"""

from typing import List

import gymnasium
import numpy as np

from bbi.agents import BaseQAgent
from bbi.environments import GoRight
from bbi.models import ExpectationModel, SamplingModel

# logger = logging.getLogger(__name__)
# logging.basicConfig(filename=f'logs/unselected_{datetime.now().strftime("%d%m%Y_%H:%M:%S")}.log', level=logging.INFO)


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
        model_type: str = "Perfect",
    ):
        super().__init__(
            action_space,
            gamma,
            environment_length,
            intensities,
            num_prize_indicators,
            initial_value,
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

        # logger.info(f'Model type: {model_type}')

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

        # Simulate forward for up to max_horizon steps
        rewards = [reward]
        simulated_state = next_state.copy()
        terminated = done
        truncated = False

        max_future_values = [
            self.get_max_future_q(simulated_state, terminated or truncated)
        ]

        next_pos, next_intensity, next_prize = self.round_obs(next_state)
        self.dynamics_model.set_state_from_rounded(
            state=[next_pos, next_intensity, next_prize], previous_status=intensity
        )

        # logger.info(f'----- Update begins -----')
        # logger.info(f"Horizon 0\tState: {[pos, intensity, prize]}\tAction: {action}\tReward: {reward}\tNext State: {[next_pos, next_intensity, next_prize]}\tFuture Q: {max_future_values[-1]}")

        for h in range(1, max_horizon + 1):
            if terminated or truncated:
                break
            action_h = self.get_action(simulated_state, greedy=True)
            next_simulated_state, reward_h, terminated, truncated, _ = (
                self.dynamics_model.step(action_h)
            )
            rewards.append(reward_h)

            max_future_values.append(
                self.get_max_future_q(next_simulated_state, terminated or truncated)
            )

            # _pos, _intensity, _prize = self.round_obs(simulated_state)
            # next_pos, next_intensity, next_prize = self.round_obs(next_simulated_state)
            # logger.info(f"Horizon {h}\tState: {[_pos, _intensity, _prize]}\tAction: {action_h}\tReward: {reward_h}\tNext State: {[next_pos, next_intensity, next_prize]}\tFuture Q: {max_future_values[-1]}")

            simulated_state = next_simulated_state.copy()

        # Compute TD targets
        td_targets = []
        cumulative_reward = 0.0
        for h, reward_h in enumerate(rewards):
            cumulative_reward += (self.gamma**h) * reward_h
            td_target = (
                cumulative_reward + (self.gamma ** (h + 1)) * max_future_values[h]
            )
            td_targets.append(td_target)

            # logger.info(f"Target {h}: {td_target}\tRewad: {reward_h}\tDiscount: {self.gamma**h}\tCumulative Reward: {cumulative_reward}\tNext Discount: {self.gamma ** (h+1)}\tFuture Q Value: {max_future_values[h]}")

        # Unselective = equal weighting
        weighted_td_target = np.mean(td_targets)
        td_error = weighted_td_target - self.q_values[pos, intensity, prize, action]
        self.q_values[pos, intensity, prize, action] += alpha * td_error
        self.td_error.append(td_error)

        # logger.info(f"Weighted Target: {weighted_td_target}")
        # logger.info(f"TD Error: {td_error}")

        return td_error
