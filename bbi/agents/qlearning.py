"""Q-Learning Agent Implementation"""

import numpy as np

from bbi.agents import BaseQAgent


class QLearningAgent(BaseQAgent):
    """A Q-Learning agent that uses a tabular 1-step update (no model, no horizon, no tau)."""

    def update_q_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        alpha: float,
        done: bool = False,
        **kwargs,
    ) -> float:
        pos, intensity, prize = self.round_obs(state)

        # 1-step TD update:
        max_future_q = self.get_max_future_q(next_state, done)
        td_target = reward + self.gamma * max_future_q * (0.0 if done else 1.0)
        td_error = td_target - self.q_values[pos, intensity, prize, action]
        self.q_values[pos, intensity, prize, action] += alpha * td_error
        self.td_error.append(td_error)

        if self.debug:
            print(
                {
                    "function": "update_q_values",
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "td_target": td_target,
                    "td_error": td_error,
                }
            )
        return td_error
