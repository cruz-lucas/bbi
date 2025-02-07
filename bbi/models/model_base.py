from typing import Dict, Tuple, Union

import numpy as np
from goright.env import GoRight

ObsType = Dict[str, Union[np.float32, np.ndarray]]


class ModelBase(GoRight):
    def predict(self, obs: ObsType, action: int) -> Tuple[ObsType, np.float32]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update(
        self, obs: ObsType, action: int, next_obs: ObsType, reward: np.float32
    ) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")
