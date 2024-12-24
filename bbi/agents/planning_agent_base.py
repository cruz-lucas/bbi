"""Base class for planning agents."""

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
            cumulative_reward += (self.gamma**h) * reward_h
            td_target = (
                cumulative_reward + (self.gamma ** (h + 1)) * max_future_values[h]
            )
            td_targets.append(td_target)
        return td_targets

    def set_model_state(self, state: np.ndarray, previous_state: np.ndarray):
        """Synchronizes the model's internal state to match the environment's discrete representation.

        Args:
            state (np.ndarray): The current continuous or rounded state.
            previous_state (np.ndarray): The previous state to determine transitions or status changes.
        """
        pos, intensity, prize = self.round_obs(state)
        _, previous_intensity, _ = self.round_obs(previous_state)

        self.dynamics_model.set_state_from_rounded(
            state=[pos, intensity, prize], previous_status=previous_intensity
        )

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
