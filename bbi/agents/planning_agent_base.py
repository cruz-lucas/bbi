from typing import List, Tuple

import numpy as np

from bbi.agents import BaseQAgent


class PlanningAgentBase(BaseQAgent):
    """Base class for planning agents that perform multi-step lookaheads using a dynamics model.

    This class:
    - Performs multi-step simulations to generate rewards and future Q-values.
    - Computes a list of TD targets.
    - Leaves weight computation (and potentially bounding box logic) to subclasses.
    """

    def simulate_rollout(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        max_horizon: int,
        done: bool,
    ) -> Tuple[List[float], List[float]]:
        """
        Performs a multi-step rollout using self.dynamics_model starting from the given next_state.
        Returns:
            rewards: A list of rewards collected from each step (including the initial reward).
            max_future_values: A list of maximum future Q-values for the states encountered.
        """
        rewards = [reward]
        simulated_state = next_state.copy()
        terminated = done
        truncated = False

        max_future_values = [
            self.get_max_future_q(simulated_state, terminated or truncated)
        ]
        self.set_model_state(simulated_state, state)

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
            simulated_state = next_simulated_state.copy()

        return rewards, max_future_values

    def compute_td_targets(
        self,
        rewards: List[float],
        max_future_values: List[float],
    ) -> List[float]:
        """
        Compute TD targets for each horizon step.
        """
        td_targets = []
        cumulative_reward = 0.0
        for h, reward_h in enumerate(rewards):
            cumulative_reward += (self.gamma**h) * reward_h
            td_target = (
                cumulative_reward + (self.gamma ** (h + 1)) * max_future_values[h]
            )
            td_targets.append(td_target)
        return td_targets

    def set_model_state(self, state: np.ndarray, previous_state: np.ndarray):
        pos, intensity, prize = self.round_obs(state)
        _, previous_intensity, _ = self.round_obs(previous_state)

        self.dynamics_model.set_state_from_rounded(
            state=[pos, intensity, prize], previous_status=previous_intensity
        )

    def compute_weights(self, td_targets: List[float], **kwargs) -> np.ndarray:
        """
        Compute the weights for the TD targets.
        This is intentionally left as a stub to be implemented by subclasses.
        """
        raise NotImplementedError

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

        rewards, max_future_values = self.simulate_rollout(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            max_horizon=max_horizon,
            done=done,
        )

        td_targets = self.compute_td_targets(rewards, max_future_values)
        weights = self.compute_weights(td_targets, **kwargs)
        weighted_td_target = np.dot(weights, td_targets)

        td_error = weighted_td_target - self.q_values[pos, intensity, prize, action]
        self.q_values[pos, intensity, prize, action] += alpha * td_error
        self.td_error.append(td_error)

        return td_error
