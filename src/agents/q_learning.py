"""Q-learning agent."""

import numpy as np
from typing import Optional

class QLearning:
    def __init__(
        self,
        learning_rate: float,
        discount_factor: float = 0.9,
        env_action_space=None,
        environment_length: int = 11,
        intensities_length: int = 3,
        num_prize_indicators: int = 2,
        rollout_length: Optional[int] = 5
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.action_space = env_action_space
        self.environment_length = environment_length
        self.num_prize_indicators = num_prize_indicators
        self.q_values = np.zeros((environment_length, intensities_length, 2**num_prize_indicators, env_action_space.n))  # positions x intensities x prize_indicator x actions
        self.td_error = []
        self.rollout_length = rollout_length
    
    def get_action(self, obs: np.ndarray, greedy: bool) -> int:

        pos, inten, prize, _, _, _ = self.round_obs(obs, obs)
        if not greedy:
            return self.action_space.sample()
        
        q_values = self.q_values[pos, inten, prize]
        ties = np.argwhere(q_values == q_values.max()).flatten()
        return np.random.choice(ties)
    

    def round_obs(self, obs, next_obs):
        # Process current observation
        pos = int(round(obs[0]))
        pos = max(0, min(pos, self.environment_length - 1))

        intensity_value = round(obs[1])
        if intensity_value <= 2.5:
            inten = 0
        elif intensity_value <= 7.5:
            inten = 1
        else:
            inten = 2

        prize = sum(int(round(obs[2 + i])) * (2 ** i) for i in range(self.num_prize_indicators))
        prize = max(0, min(prize, 2 ** self.num_prize_indicators - 1))

        # Process next observation
        next_pos = int(round(next_obs[0]))
        next_pos = max(0, min(next_pos, self.environment_length - 1))

        next_intensity_value = round(next_obs[1])
        if next_intensity_value <= 2.5:
            next_inten = 0
        elif next_intensity_value <= 7.5:
            next_inten = 1
        else:
            next_inten = 2

        next_prize = sum(int(round(next_obs[2 + i])) * (2 ** i) for i in range(self.num_prize_indicators))
        next_prize = max(0, min(next_prize, 2 ** self.num_prize_indicators - 1))

        return pos, inten, prize, next_pos, next_inten, next_prize



    def update_q_values(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        use_value_expansion: bool = False,
        dynamics_model: Optional[callable] = None,
    ):
        pos, inten, prize, next_pos, next_inten, next_prize = self.round_obs(obs, next_obs)        
        future_q = np.max(self.q_values[next_pos, next_inten, next_prize])
        td_target = reward + self.discount_factor * future_q

        # if use_value_expansion:          
        #     current_state = obs.copy()
        #     returns = 0.0

        #     for t in range(2, self.rollout_length+1):
        #         action_h = self.get_action(current_state, greedy=True)
        #         # Simulate next state and reward
        #         next_state, reward_h, terminated, truncated, info = dynamics_model.step(action_h)
        #         pos, inten, prize, next_pos, next_inten, next_prize = self.round_obs(obs, next_obs)

        #         future_q = np.max(self.q_values[next_pos, next_inten, next_prize])
        #         returns += self.discount_factor ** t * reward_h
        #         td_target += reward + returns + self.discount_factor ** t * future_q
                
        #         current_state = next_state

        #     td_target = td_target/self.rollout_length

        td_error = td_target - self.q_values[pos, inten, prize, action]
        self.q_values[pos, inten, prize, action] += self.lr * td_error
        self.td_error.append(td_error)

