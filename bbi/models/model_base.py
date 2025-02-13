"""Base class for models."""

from typing import Dict, Tuple, Union

import numpy as np
from goright.env import GoRight

ObsType = Dict[str, Union[np.float32, np.ndarray]]


class ModelBase(GoRight):
    """_summary_.

    Args:
        GoRight (_type_): _description_
    """

    def predict(
        self,
        obs: Tuple[int, ...],
        action: int,
        **kwargs,
    ) -> (
        Tuple[Tuple[int, ...], float]
        | Tuple[Tuple[int, ...], float, Tuple[int, ...], float, Tuple[int, ...], float]
    ):
        """_summary_.

        Args:
            self (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update(
        self,
        obs: Tuple[int, ...],
        action: int,
        next_obs: Tuple[int, ...],
        reward: np.float32,
    ) -> None:
        """_summary_.

        Args:
            obs (Tuple[int, ...]): _description_
            action (int): _description_
            next_obs (Tuple[int, ...]): _description_
            reward (np.float32): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
