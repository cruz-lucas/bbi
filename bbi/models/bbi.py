"""Module with BBI model."""

from typing import List, Optional, Tuple

import numpy as np

from .expectation import ExpectationModel


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
        has_state_offset: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[int] = "human",
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
        self.state_bounding_box = None
        self.reward_bounding_box = None

        self.LOWER_BOUND = 0
        self.UPPER_BOUND = 1

        # rendering related
        self.bottom_area_height = 150

    def reset(self, seed=None, options=None):
        self.state_bounding_box = None
        self.reward_bounding_box = None
        return super().reset(seed, options)

    def get_next_bounds(
        self, state: np.ndarray, action_bounds: np.ndarray | List[int] = [0, 1]
    ) -> Tuple[np.ndarray, np.ndarray]:
        position, current_status, *prize_indicators = state
        prize_indicators = np.array(prize_indicators)
        if self.state_bounding_box is None:
            previous_prize_indicators_list = [prize_indicators]

        else:
            previous_prize_indicators_list = [
                self.state_bounding_box[0, 2:],
                self.state_bounding_box[1, 2:],
            ]

        status_bounds = [np.min(self.intensities), np.max(self.intensities)]
        next_positions = []
        prize_indicators_list = []

        for action in action_bounds:
            next_pos = self._compute_next_position(action=action, position=position)
            next_positions.append(next_pos)

            for status in status_bounds:
                for prize_ind in previous_prize_indicators_list:
                    prize_indicators_list.append(
                        self._compute_next_prize_indicators(
                            next_pos, position, status, prize_ind
                        )
                    )

        prize_indicators_bounds = [
            np.min(prize_indicators_list, axis=0),
            np.max(prize_indicators_list, axis=0),
        ]
        next_pos_bounds = [np.min(next_positions), np.max(next_positions)]

        state_bounds = np.zeros((2, 2 + self.num_prize_indicators))
        state_bounds[self.LOWER_BOUND, 0] = next_pos_bounds[self.LOWER_BOUND]
        state_bounds[self.LOWER_BOUND, 1] = status_bounds[self.LOWER_BOUND]

        state_bounds[self.UPPER_BOUND, 0] = next_pos_bounds[self.UPPER_BOUND]
        state_bounds[self.UPPER_BOUND, 1] = status_bounds[self.UPPER_BOUND]

        for i in range(self.num_prize_indicators):
            p_min = prize_indicators_bounds[self.LOWER_BOUND][i]
            p_max = prize_indicators_bounds[self.UPPER_BOUND][i]
            state_bounds[self.LOWER_BOUND, 2 + i] = p_min
            state_bounds[self.UPPER_BOUND, 2 + i] = p_max

        self.state_bounding_box = state_bounds
        self.reward_bounding_box = [
            self._compute_reward(prize_indicators_bounds[0], action, position),
            self._compute_reward(prize_indicators_bounds[1], action, position),
        ]

        return self.state_bounding_box, self.reward_bounding_box

    # def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    #     position, current_status, *prize_indicators = self.state
    #     prize_indicators = np.array(prize_indicators)

    #     self.state, reward, terminated, truncated, _ = self._step(action)

    #     if self.state_bounding_box is None:
    #         prize_indicators_min = prize_indicators
    #         prize_indicators_max = prize_indicators

    #     else:
    #         prize_indicators_min = self.state_bounding_box[0, 2:]
    #         prize_indicators_max = self.state_bounding_box[1, 2:]

    #     status_min = np.min(self.intensities)
    #     status_max = np.max(self.intensities)
    #     next_pos = self.state[0]

    #     prize_indicators_min = self._compute_next_prize_indicators(
    #         next_pos, position, status_min, prize_indicators_min
    #     )
    #     prize_indicators_max = self._compute_next_prize_indicators(
    #         next_pos, position, status_max, prize_indicators_max
    #     )

    #     reward_min = self._compute_reward(prize_indicators_min, action, position)
    #     reward_max = self._compute_reward(prize_indicators_max, action, position)

    #     state_bounding_box = np.zeros((2, 2 + self.num_prize_indicators))
    #     state_bounding_box[0, 0] = next_pos
    #     state_bounding_box[1, 0] = next_pos
    #     state_bounding_box[0, 1] = status_min
    #     state_bounding_box[1, 1] = status_max
    #     for i in range(self.num_prize_indicators):
    #         p_min = prize_indicators_min[i]
    #         p_max = prize_indicators_max[i]
    #         state_bounding_box[0, 2 + i] = p_min
    #         state_bounding_box[1, 2 + i] = p_max

    #     self.state_bounding_box = state_bounding_box
    #     self.reward_bounding_box = [reward_min, reward_max]

    #     return self.state, reward, terminated, truncated, {}
