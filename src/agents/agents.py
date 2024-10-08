"""This module implements a Q-Learning agent using a tabular approach."""

from abc import ABC
from typing import Tuple
from typing import Optional

import gymnasium
import numpy as np

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    filename=f'logs/agents_{datetime.now().strftime("%d%m%Y_%H%M%S")}.log',
)


class QLearningAgent(ABC):
    """A Q-Learning agent that learns Q-values for state-action pairs using temporal difference learning.

    Attributes:
        action_space (gymnasium.Space): The action space of the environment.
        dynamics_model (gymnasium.Env): The dynamics model used to simulate environment transitions.
        gamma (float): The discount factor for future rewards.
        max_horizon (int): The maximum horizon for multi-step updates.
        environment_length (int): The length of the environment's state representation.
        num_prize_indicators (int): The number of prize indicators in the state.
        q_values (np.ndarray): The Q-table storing state-action values.
    """

    def __init__(
        self,
        action_space: gymnasium.Space,        
        gamma: float = 0.9,
        max_horizon: int = 5,
        environment_length: int = 11,
        intensities_length: int = 3,
        num_prize_indicators: int = 2,
        initial_value: int = 0,
    ) -> None:
        """
        Initializes the QLearningAgent with the given parameters.

        Args:
            action_space (gymnasium.Space): The action space of the environment.
            dynamics_model (gymnasium.Env): The dynamics model used to simulate environment transitions.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.
            max_horizon (int, optional): The maximum horizon for multi-step updates. Defaults to 5.
            environment_length (int, optional): The length of the environment's state representation. Defaults to 11.
            intensities_length (int, optional): The number of intensity levels. Defaults to 3.
            num_prize_indicators (int, optional): The number of prize indicators in the state. Defaults to 2.
        """
        self.action_space = action_space
        self.gamma = gamma
        self.max_horizon = max_horizon
        self.environment_length = environment_length
        self.num_prize_indicators = num_prize_indicators
        self.q_values = (
            np.zeros(
                (
                    environment_length,
                    intensities_length,
                    2**num_prize_indicators,
                    action_space.n,
                ),
                dtype=float,
            )
            + initial_value
        )

        self.td_error = []
        self.td_error_matrix = []

        self.debug = True

    def round_obs(self, obs: np.ndarray) -> Tuple[int, int, int]:
        """Rounds and discretizes the observation into discrete state components.

        Args:
            obs (np.ndarray): The continuous observation from the environment.

        Returns:
            Tuple[int, int, int]: A tuple containing the rounded position, intensity level, and prize indicator.
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
        """Selects an action based on the current state and exploration strategy.

        Args:
            state (np.ndarray): The current state observation.
            greedy (bool): If True, selects the action with the highest Q-value. Otherwise, selects a random action.

        Returns:
            int: The selected action.
        """
        pos, intensity, prize = self.round_obs(state)

        if not greedy:
            return self.action_space.sample()

        q_values = self.q_values[pos, intensity, prize]
        ties = np.argwhere(q_values == q_values.max()).flatten()

        if self.debug:
            log_dict = {
                "function": "get_action",
                "state": state,
                "q_values": self.q_values[pos, intensity, prize],
                "ties": ties,
            }
            logging.debug(log_dict)
    
        return int(np.random.choice(ties))

    def update_q_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        alpha: float,
        tau: float,
        dynamics_model: Optional[gymnasium.Env] = None,
    ) -> None:
        """Updates the Q-values for a given state-action pair using multi-step temporal difference errors.

        Args:
            state (np.ndarray): The current state observation.
            action (int): The action taken.
            reward (float): The reward received after taking the action.
            next_state (np.ndarray): The next state observation after taking the action.
            alpha (float): The learning rate.
            tau (float): The temperature parameter for weighting TD targets.
        """
        pos, intensity, prize = self.round_obs(state)

        td_targets = []
        for horizon in range(1, self.max_horizon + 1):
            td_target = self.calculate_td_target(
                reward=reward,
                next_state=next_state,
                horizon=horizon,
                dynamics_model=dynamics_model
            )
            td_targets.append(td_target)

        weights = self.calculate_weights(state, action, td_targets, tau)
        weighted_td_target = sum(
            weights[i] * td_targets[i] for i in range(len(td_targets))
        )

        td_error = weighted_td_target - self.q_values[pos, intensity, prize, action]
        self.q_values[pos, intensity, prize, action] += alpha * td_error
        self.td_error.append(td_error)

        if self.debug:
            log_dict = {
                "function": "update_q_values",
                "state": state,
                "action": action,
                "next_state": next_state,
                "targets": td_targets,
                "weighted_td_target": weighted_td_target,
                "q_values": self.q_values[pos, intensity, prize],
            }
            logging.debug(log_dict)

        return td_error

    def calculate_td_target(
        self,
        reward: float,
        next_state: np.ndarray,
        horizon: int,
        dynamics_model: Optional[gymnasium.Env] = None,
    ) -> float:
        """Calculates the temporal difference (TD) target for a specific horizon.

        Args:
            reward (float): The immediate reward received.
            current_state (np.ndarray): The current state observation.
            next_state (np.ndarray): The next state observation after taking an action.
            horizon (int): The number of steps ahead to consider for the TD target.

        Returns:
            float: The calculated TD target.
        """
        total_discounted_return = reward
        discount = self.gamma
        simulated_state = next_state.copy()

        for h in range(1, horizon):
            action = self.get_action(simulated_state, greedy=True)
            next_simulated_state, reward, terminated, truncated, _ = dynamics_model.step(action)

            total_discounted_return += discount * reward
            discount *= self.gamma

            if terminated or truncated:
                break
            
            simulated_state = next_simulated_state

        pos, intensity, prize = self.round_obs(simulated_state)
        max_future_q = np.max(self.q_values[pos, intensity, prize])

        target = total_discounted_return + discount * max_future_q
        return target

    def calculate_weights(
        self,
        state: np.ndarray,
        action: int,
        td_targets: list[float],
        tau: float,
    ) -> np.ndarray:
        """Calculates weights for each TD target based on uncertainty.

        Args:
            state (np.ndarray): The current state observation.
            action (int): The action taken.
            td_targets (list of float): The list of TD targets for different horizons.
            tau (float): The temperature parameter for weighting.

        Returns:
            np.ndarray: An array of weights corresponding to each TD target.
        """
        # Placeholder for uncertainty-based weighting
        # To be implemented based on the dynamics model's uncertainty estimates
        weights = np.ones(len(td_targets)) / len(td_targets)

        assert np.isclose(a=weights.sum(), b=1.0, atol=1e-5)

        return weights
