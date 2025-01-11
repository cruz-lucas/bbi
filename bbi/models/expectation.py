"""Module with expectation model."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bbi.environments import GoRight
from bbi.utils.dataclasses import State


class ExpectationModel(GoRight):
    metadata = {"render_modes": ["human"], "environment_name": "Expectation Model"}

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        seed: Optional[int] = None,
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
        super().reset(seed=seed)

        if self.state is None:
            raise ValueError("State has not been initialized.")
        self.state.current_status_indicator = 5

        return self.state.get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """_summary_

        Args:
            action (int): _description_

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: _description_
        """
        return self._step(action)

    def _step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """_summary_

        Args:
            action (int): _description_

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: _description_
        """
        if self.state is None:
            raise ValueError("State has not been initialized.")

        current_state: State = self.state

        next_pos = self._compute_next_position(action, current_state)
        next_status = 5
        next_prize_indicators = self._compute_expected_next_prize_indicators(
            next_pos, current_state
        )

        reward = self._compute_reward(next_prize_indicators, action, current_state)

        self.state = State(
            position=next_pos,
            previous_status_indicator=current_state.current_status_indicator,
            current_status_indicator=next_status,
            prize_indicators=next_prize_indicators,
        )

        self.tracker.record(
            state=current_state, action=action, reward=reward, next_state=self.state
        )

        return self.state.get_observation(), reward, False, False, {}

    def _compute_expected_next_prize_indicators(
        self,
        next_position: float,
        current_state: State,
    ) -> np.ndarray:
        """Computes the next prize indicators based on the current state.

        Args:
            next_position (float): _description_
            position (float): _description_
            next_status (int): _description_
            prize_indicators (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        position, _, _, *prize_indicators = current_state.get_state()

        prize_indicators = np.array(prize_indicators)
        if int(next_position) == self.length - 1:
            if int(position) == self.length - 2:
                return np.ones_like(prize_indicators, dtype=float) / 3.0
            elif all(prize_indicators == 1):
                return prize_indicators
            else:
                return self._shift_prize_indicators(prize_indicators)
        return np.zeros(self.num_prize_indicators, dtype=int)
