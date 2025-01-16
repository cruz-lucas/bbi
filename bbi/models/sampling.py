"""Module with sampling model."""

from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np

from bbi.environments import GoRight
from bbi.utils.dataclasses import State


class SamplingModel(GoRight):
    metadata = {"render_modes": ["human"], "environment_name": "Sampling Model"}

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        seed: Optional[int] = None,
    ) -> None:
        """_summary_

        Args:
            num_prize_indicators (int, optional): _description_. Defaults to 2.
            env_length (int, optional): _description_. Defaults to 11.
            status_intensities (List[int], optional): _description_. Defaults to [0, 5, 10].
            seed (Optional[int], optional): _description_. Defaults to None.
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
        """_summary_

        Args:
            seed (Optional[int], optional): _description_. Defaults to None.
            options (Optional[Dict[str, Any]], optional): _description_. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: _description_
        """
        super().reset(seed=seed)

        if self.state is None:
            raise ValueError("State has not been initialized.")

        return self.state.get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """_summary_

        Args:
            action (int): _description_

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: _description_
        """
        if self.state is None:
            raise ValueError("State has not been initialized.")

        current_state: State = deepcopy(self.state)

        next_pos = self._compute_next_position(action)
        next_status = np.random.choice(self.intensities)
        next_prize_indicators = self._compute_next_prize_indicators(
            next_pos
        )

        reward = self._compute_reward(next_prize_indicators, action)

        self.state.set_state(
            position=next_pos,
            previous_status_indicator=current_state.current_status_indicator,
            current_status_indicator=next_status,
            prize_indicators=next_prize_indicators,
        )

        self.tracker.record(
            state=current_state, action=action, reward=reward, next_state=self.state
        )

        return self.state.get_observation(), reward, False, False, {}

    def _compute_next_prize_indicators(
        self, next_position: float
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
