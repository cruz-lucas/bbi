"""Module containing all environments."""

from gymnasium.envs.registration import register

from bbi.environments.base_env import BaseEnv
from bbi.environments.goright import GoRight

ENV_CONFIGURATION = {
    "bbi/goRight-v0": {
        "has_state_offset": True,
        "env_length": 11,
        "num_prize_indicators": 2,
        "status_intensities": [0, 5, 10],
    }
}

env_id = "bbi/goRight-v0"
register(
    id=env_id,
    entry_point="bbi.environments.goright:GoRight",
    kwargs=ENV_CONFIGURATION.get(env_id),
)

__all__ = ["GoRight", "BaseEnv"]
