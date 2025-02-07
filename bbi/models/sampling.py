"""Module with sampling model."""

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bbi.models.model_base import ModelBase, ObsType


class ExpectationModel(ModelBase):
    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initializes the GoRight environment.

        Args:
            num_prize_indicators (int): Number of prize indicators.
            env_length (int): Length of the grid.
            status_intensities (List[int]): Possible status intensities.
            has_state_offset (bool): Whether to add noise to observations.
            seed (Optional[int]): Seed for reproducibility.
        """
        super().__init__(
            num_prize_indicators=num_prize_indicators,
            env_length=env_length,
            status_intensities=status_intensities,
            has_state_offset=False,
            seed=seed,
            render_mode=render_mode,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to its initial state.

        Args:
            seed (Optional[int]): Seed for reproducibility.
            options (Optional[Dict[str, Any]]): Additional options.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info dictionary.
        """
        obs, info = super().reset(seed=seed)

        self.state.current_status_indicator = self._compute_next_status()

        return obs, info

    def _compute_next_status(self, **kwargs) -> int:
        """Returns the expected next status.

        Returns:
            int: Expected ned status.
        """
        return np.random.choice(self.status_intensities)

    def _compute_next_prize_indicators(self, next_position: float) -> np.ndarray:
        """Computes the next prize indicators based on the current state.

        Args:
            next_position (float): _description_
            position (float): _description_
            next_status (int): _description_
            prize_indicators (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        if int(next_position) == self.length - 1:
            if int(self.state.position) == self.length - 2:
                return np.random.choice(
                    [0, 1], size=self.num_prize_indicators, p=[2 / 3, 1 / 3]
                )
            elif all(self.state.prize_indicators == 1):
                return self.state.prize_indicators
            else:
                return self._shift_prize_indicators(self.state.prize_indicators)
        return np.zeros(self.num_prize_indicators, dtype=int)

    def predict(
        self,
        obs: Tuple[int, ...],
        action: int,
        state_bb: Tuple[Tuple[int, ...], Tuple[int, ...]] | None = None,
        action_bb: Tuple[int, int] | None = None,
    ) -> Tuple[ObsType, np.float32, ObsType, np.float32, ObsType, np.float32]:
        """Predict the next state and reward given an observation and action, and also compute lower and upper bounds by using the minimum and maximum status intensities, respectively.

        This method sets the environment’s state based on the observation,
        calls step() to simulate the transition, then repeats the process with the
        status indicator forced to its minimum and maximum values. The environment’s
        internal state is restored at the end.

        Args:
            obs: A dictionary with keys 'position', 'status_indicator', and 'prize_indicators'
                representing the current observation.
            action: The action to be executed (e.g. 0 for LEFT, 1 for RIGHT).

        Returns:
            A tuple containing:
            - expected_obs: Expected next observation.
            - expected_reward: Expected reward.
            - lower_obs: Next observation when the status is forced to the minimum.
            - lower_reward: Reward for the lower bound.
            - upper_obs: Next observation when the status is forced to the maximum.
            - upper_reward: Reward for the upper bound.
        """
        pos = obs[0]  # round(obs["position"][0])
        status = obs[1]  # round(obs["status_indicator"][0])
        prize = np.array(obs[2:])  # np.array(obs["prize_indicators"])

        if status == 1:
            status = 5
        elif status == 2:
            status = 10

        self.state.set_state(
            position=pos,
            current_status_indicator=status,
            prize_indicators=np.array(prize),
        )

        self._compute_next_status = lambda *args, **kwargs: np.random.choice(
            self.status_intensities
        )
        _, exp_reward, _, _, _ = self.step(action)
        exp_obs = self.state.get_state()[self.state.mask]

        if state_bb is not None:
            state_ranges = [
                range(state_bb[0][i], state_bb[1][i] + 1)
                for i in range(len(state_bb[0]))
            ]
            action_range = range(action_bb[0], action_bb[1] + 1)

            state_candidates = []
            reward_candidates = []
            for s in product(*state_ranges):
                for a in action_range:
                    self.state.set_state(
                        position=pos,
                        current_status_indicator=status,
                        prize_indicators=np.array(prize),
                    )

                    self._compute_next_status = lambda *args, **kwargs: 0
                    _, lower_reward, _, _, _ = self.step(action)
                    state_candidates.append(self.state.get_state()[self.state.mask])
                    reward_candidates.append(lower_reward)

                    self.state.set_state(
                        position=pos,
                        current_status_indicator=status,
                        prize_indicators=np.array(prize),
                    )

                    self._compute_next_status = lambda *args, **kwargs: 10
                    _, upper_reward, _, _, _ = self.step(action)
                    state_candidates.append(self.state.get_state()[self.state.mask])
                    reward_candidates.append(upper_reward)

            lower_reward = np.min(reward_candidates)
            upper_reward = np.max(reward_candidates)

            lower_obs = np.min(state_candidates, axis=0)
            upper_obs = np.max(state_candidates, axis=0)
            lower_obs[1] = np.argwhere(self.status_intensities == lower_obs[1])
            upper_obs[1] = np.argwhere(self.status_intensities == upper_obs[1])
            exp_obs[1] = np.argwhere(self.status_intensities == exp_obs[1])

            self._compute_next_status = lambda *args, **kwargs: np.random.choice(
                self.status_intensities
            )
            return (
                tuple(exp_obs),
                exp_reward,
                tuple(lower_obs),
                lower_reward,
                tuple(upper_obs),
                upper_reward,
            )

        return (exp_obs, exp_reward)

    def update(self, **kwargs) -> None:
        return None
