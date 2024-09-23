# q_learning.py

import numpy as np
from typing import Optional
import random

class QLearning:
    def __init__(
        self,
        learning_rate: float,
        discount_factor: float = 0.9,
        env_action_space=None,
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.action_space = env_action_space
        self.q_values = np.zeros((10, 3, 4, env_action_space.n))  # positions x intensities x prize_indicator x actions
        self.td_error = []

    def get_action(self, obs: np.ndarray, greedy: bool) -> int:
        position = int(round(obs[0]))
        position = max(0, min(position, 9))  # Ensure position is within [0, 9]

        intensity_value = int(round(obs[1]))
        if intensity_value <= 2.5:
            intensity = 0
        elif intensity_value <= 7.5:
            intensity = 1
        else:
            intensity = 2

        prize_indicator = int(round(obs[2])) * 2 + int(round(obs[3]))
        prize_indicator = max(0, min(prize_indicator, 3))  # Ensure prize_indicator is within [0, 3]

        if not greedy:
            return random.randint(0, self.action_space.n - 1)
        q_values = self.q_values[position, intensity, prize_indicator]
        return int(np.argmax(q_values))

    def update_q_values(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
    ):
        # Process current observation
        pos = int(round(obs[0]))
        pos = max(0, min(pos, 9))

        intensity_value = int(round(obs[1]))
        if intensity_value <= 2.5:
            inten = 0
        elif intensity_value <= 7.5:
            inten = 1
        else:
            inten = 2

        prize = int(round(obs[2])) * 2 + int(round(obs[3]))
        prize = max(0, min(prize, 3))

        # Process next observation
        next_pos = int(round(next_obs[0]))
        next_pos = max(0, min(next_pos, 9))

        next_intensity_value = int(round(next_obs[1]))
        if next_intensity_value <= 2.5:
            next_inten = 0
        elif next_intensity_value <= 7.5:
            next_inten = 1
        else:
            next_inten = 2

        next_prize = int(round(next_obs[2])) * 2 + int(round(next_obs[3]))
        next_prize = max(0, min(next_prize, 3))

        future_q = np.max(self.q_values[next_pos, next_inten, next_prize])
        td_target = reward + self.discount_factor * future_q
        td_error = td_target - self.q_values[pos, inten, prize, action]
        self.q_values[pos, inten, prize, action] += self.lr * td_error
        self.td_error.append(td_error)
