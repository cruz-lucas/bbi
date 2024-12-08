"""Module implementing a tabular Q-Learning agent."""

from typing import List, Optional, Tuple

import gymnasium
import numpy as np


class QLearningAgent:
    """A Q-Learning agent that uses a tabular approach for state-action value estimation.

    Attributes:
        action_space (gymnasium.Space): The action space of the environment.
        gamma (float): Discount factor for future rewards.
        environment_length (int): Length of the environment's state representation.
        num_prize_indicators (int): Number of prize indicators in the state.
        q_values (np.ndarray): Q-table storing state-action values.
    """

    def __init__(
        self,
        action_space: gymnasium.Space,
        gamma: float = 0.9,
        environment_length: int = 11,
        intensities_length: int = 3,
        num_prize_indicators: int = 2,
        initial_value: float = 0.0,
    ) -> None:
        """Initializes the QLearningAgent with the given parameters.

        Args:
            action_space (gymnasium.Space): The action space of the environment.
            gamma (float, optional): Discount factor for future rewards.
            environment_length (int, optional): Length of the environment's state representation.
            intensities_length (int, optional): Number of intensity levels.
            num_prize_indicators (int, optional): Number of prize indicators in the state.
            initial_value (float, optional): Initial value for Q-table entries.
        """
        self.action_space = action_space
        self.gamma = gamma
        self.environment_length = environment_length
        self.num_prize_indicators = num_prize_indicators
        self.q_values = np.full(
            (
                environment_length,
                intensities_length,
                2**num_prize_indicators,
                action_space.n,
            ),
            fill_value=initial_value,
            dtype=float,
        )
        self.td_error: List[float] = []
        self.debug: bool = False

    def round_obs(self, obs: np.ndarray) -> Tuple[int, int, int]:
        """Discretizes the continuous observation into discrete state components.

        Args:
            obs (np.ndarray): The continuous observation from the environment.

        Returns:
            Tuple[int, int, int]: Discrete (position, intensity level, prize indicator).
        """
        if obs[1] <= 2.5:
            intensity_value = 0
        elif obs[1] <= 7.5:
            intensity_value = 1
        else:
            intensity_value = 2

        prize_indicator = sum(
            int(obs[2 + i] >= 0.5) * (2**i) for i in range(self.num_prize_indicators)
        )

        return (
            int(round(obs[0])),
            intensity_value,
            prize_indicator,
        )

    def get_action(self, state: np.ndarray, greedy: bool) -> int:
        """Selects an action based on the current state.

        Args:
            state (np.ndarray): The current state observation.
            greedy (bool): If True, selects the action with the highest Q-value; else random.

        Returns:
            int: The selected action.
        """
        pos, intensity, prize = self.round_obs(state)

        if not greedy:
            return self.action_space.sample()

        q_values = self.q_values[pos, intensity, prize]
        max_value = np.max(q_values)
        ties = np.flatnonzero(q_values == max_value)

        if self.debug:
            print(
                {
                    "function": "get_action",
                    "state": state,
                    "q_values": q_values,
                    "ties": ties,
                }
            )

        return int(np.random.choice(ties))

    def update_q_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        alpha: float,
        tau: float,
        max_horizon: int = 5,
        dynamics_model: Optional[gymnasium.Env] = None,
    ) -> float:
        """Updates the Q-values for a given state-action pair.

        Args:
            state (np.ndarray): The current state observation.
            action (int): The action taken.
            reward (float): The immediate reward received.
            next_state (np.ndarray): The next state observation.
            alpha (float): The learning rate.
            tau (float): Temperature parameter for weighting TD targets.
            max_horizon (int, optional): Maximum horizon for multi-step updates.
            dynamics_model (Optional[gymnasium.Env], optional): Model to simulate environment transitions.

        Returns:
            float: The temporal-difference error.
        """
        pos, intensity, prize = self.round_obs(state)

        rewards = [reward]
        simulated_state = next_state.copy()
        terminated = False
        truncated = False

        max_future_values = [
            self.get_max_future_q(simulated_state, terminated or truncated)
        ]

        # Perform rollout for multi-step TD targets
        for _ in range(1, max_horizon):
            if terminated or truncated:
                break

            action_h = self.get_action(simulated_state, greedy=True)
            next_simulated_state, reward_h, terminated, truncated, _ = (
                dynamics_model.step(action_h)
            )
            rewards.append(reward_h)
            simulated_state = next_simulated_state.copy()
            max_future_q = self.get_max_future_q(
                simulated_state, terminated or truncated
            )
            max_future_values.append(max_future_q)

        # Compute TD targets
        td_targets = []
        cumulative_reward = 0.0
        for h, reward_h in enumerate(rewards):
            cumulative_reward += (self.gamma**h) * reward_h
            td_target = (
                cumulative_reward + (self.gamma ** (h + 1)) * max_future_values[h]
            )
            td_targets.append(td_target)

        weights = self._get_weights(td_targets)

        weighted_td_target = np.dot(weights, td_targets)
        td_error = weighted_td_target - self.q_values[pos, intensity, prize, action]
        self.q_values[pos, intensity, prize, action] += alpha * td_error
        self.td_error.append(td_error)

        if self.debug:
            print(
                {
                    "function": "update_q_values",
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "td_targets": td_targets,
                    "weighted_td_target": weighted_td_target,
                    "q_values": self.q_values[pos, intensity, prize],
                }
            )

        return td_error

    def _get_weights(self, td_targets):
        weights = np.ones(len(td_targets)) / len(td_targets)
        return weights

    def get_max_future_q(self, state: np.ndarray, done: bool) -> float:
        """Retrieves the maximum future Q-value for a given state.

        Args:
            state (np.ndarray): The state observation.
            done (bool): Whether the state is terminal.

        Returns:
            float: The maximum future Q-value.
        """
        if not done:
            pos, intensity, prize = self.round_obs(state)
            max_future_q = np.max(self.q_values[pos, intensity, prize])
        else:
            max_future_q = 0.0
        return max_future_q
