"""Module for base environemnt class."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env, Space
from gymnasium.utils import seeding

from bbi.utils.dataclasses import State


class BaseEnv(Env):
    def __init__(self) -> None:
        """Initializes the base environment with rendering and dimension settings."""
        super().__init__()

        self.action_space: Optional[Space] = None
        self.observation_space: Optional[Space] = None
        self.state: Optional[State] = None

    def seed(self, seed: Optional[int] = None) -> None:
        """Sets the seed for the environment's random number generator.

        Args:
            seed: The seed value.
        """
        self.np_random, _ = seeding.np_random(seed)

    def step(
        self, action: int
    ) -> Tuple[Dict[str, float | int | np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Performs one timestep transition in the environment. Must be overridden.

        Returns:
            Tuple containing next observation, reward, termination flag, truncation flag, and info dictionary."""
        raise NotImplementedError("This method should be implemented by subclasses.")
