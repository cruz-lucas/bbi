"""Module for base environemnt class."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class BaseEnv(gym.Env):
    def __init__(self) -> None:
        """Initializes the base environment with rendering and dimension settings."""
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
        """Performs one timestep transition in the environment. Must be overridden.

        Returns:
            Tuple containing next observation, reward, termination flag, truncation flag, and info dictionary."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_observation(self) -> np.ndarray:
        """Returns the current observation of the environment's state. Must be overridden.

        Returns:
            A numpy array representing the current observation."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_state(self, state: np.ndarray, previous_status: int) -> None:
        """Manually sets the environment's state. Must be overridden.

        Returns:
            None"""
        raise NotImplementedError("This method should be implemented by subclasses.")
