from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class BaseEnv(gym.Env):
    def __init__(self) -> None:
        """"""
        super().__init__()

        self.LEFT = 0
        self.RIGHT = 1

        # Rendering related attributes
        self.screen = None
        self.clock = None
        self.font = None

        self.cell_size = 60
        self.margin = 50
        self.bottom_area_height = 50
        self.top_area_height = 100
        self.grid_top = 100

        # Image sprites for lamps
        self.lamp_on = None
        self.lamp_off = None
        self.robot = None

    def seed(self, seed: Optional[int] = None) -> None:
        """Sets the seed for the environment's random number generator.

        Args:
            seed: The seed value.
        """
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Abstract method: Implement in subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_observation(self) -> np.ndarray:
        """Abstract method: Implement in subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_state(self, state: np.ndarray, previous_status: int) -> None:
        """Abstract method: Implement in subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")
