"""Module with BBI model."""

from typing import List, Optional, Tuple
from copy import deepcopy
import numpy as np

from .expectation import ExpectationModel
from bbi.utils.dataclasses import State, BoundingBox, Action, BBITracker


class BBI(ExpectationModel):
    metadata = {
        "render_modes": ["human"],
        "environment_name": "BBI",
    }

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        seed: Optional[int] = None,
        render_mode: Optional[str] = "human",
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
            seed=seed,
        )
        self.bounding_box: BoundingBox | None = None
        self.rolling_bounding_box: BoundingBox | None = None
        self.bbi_tracker: BBITracker | None = None

    def reset(self, seed=None, options=None):
        """_summary_

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        obs, info = super().reset(seed, options)
        self.bounding_box = None
        self.rolling_bounding_box = None
        self.bbi_tracker = BBITracker()

        current_state_bb = BoundingBox(
                state_lower_bound=self.state,
                state_upper_bound=self.state,
                reward_lower_bound=np.nan,
                reward_upper_bound=np.nan
            )
        self.bbi_tracker.record(deepcopy(current_state_bb))

        return obs, info
    
    def step(self, action):
        current_state_bb = BoundingBox(
                state_lower_bound=self.state,
                state_upper_bound=self.state,
                reward_lower_bound=np.nan,
                reward_upper_bound=np.nan
            )
        
        self.bounding_box = self.get_next_bounds(current_state_bb)

        if self.rolling_bounding_box is None:
            self.rolling_bounding_box = self.get_next_bounds(current_state_bb)
        else:
            self.rolling_bounding_box = self.get_next_bounds(self.rolling_bounding_box)

        self.bbi_tracker.record(deepcopy(self.rolling_bounding_box))

        return super().step(action)

    def get_next_bounds(
        self, bounding_box: BoundingBox, action_bounds: np.ndarray | List[int] = [0, 1]
    ) -> BoundingBox:
        """_summary_

        Args:
            state (np.ndarray): _description_
            action_bounds (np.ndarray | List[int], optional): _description_. Defaults to [0, 1].

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        previous_prize_indicators_list = [
            bounding_box.state_lower_bound.prize_indicators,
            bounding_box.state_upper_bound.prize_indicators
        ]

        previous_pos = [
            bounding_box.state_lower_bound.position,
            bounding_box.state_upper_bound.position
        ]

        status_bounds = [np.min(self.intensities), np.max(self.intensities)] # the bounds will always be 0 and 10

        next_positions = []
        prize_indicators_list = []

        for action in action_bounds:
            for pos in previous_pos:
                next_pos = self._compute_next_position(action=action, position=pos)
                next_positions.append(next_pos)

                for status in status_bounds:
                    for prize_ind in previous_prize_indicators_list:
                        prize_indicators_list.append(
                            self._compute_next_prize_indicators(
                                next_pos, next_status=status, prize_indicators=prize_ind
                            )
                        )

        lower_bound: State = State(
            position=np.min(next_positions),
            previous_status_indicator=np.min(self.intensities),
            current_status_indicator=np.min(self.intensities),
            prize_indicators=np.min(prize_indicators_list, axis=0)
        )

        upper_bound: State = State(
            position=np.max(next_positions),
            previous_status_indicator=np.max(self.intensities),
            current_status_indicator=np.max(self.intensities),
            prize_indicators=np.max(prize_indicators_list, axis=0)
        )

        reward_bounds = []
        for action in action_bounds:
            for indicator in [lower_bound.prize_indicators, upper_bound.prize_indicators]:
                reward_bounds.append(self._compute_reward(indicator, action))


        bounding_box = BoundingBox(
            state_lower_bound=lower_bound,
            state_upper_bound=upper_bound,
            reward_lower_bound=np.min(reward_bounds),
            reward_upper_bound=np.max(reward_bounds)
        )

        return bounding_box
