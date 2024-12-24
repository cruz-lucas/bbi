"""Unselective Planning Agent"""

from typing import List

import gymnasium
import numpy as np

from bbi.agents import PlanningAgentBase
from bbi.environments import GoRight
from bbi.models import ExpectationModel, SamplingModel


class UnselectivePlanningAgent(PlanningAgentBase):
    """An agent that performs multi-step TD updates using a dynamics model, with equal (unselective) weighting.

    Args:
        PlanningAgentBase (_type_): _description_
    """

    def __init__(
        self,
        action_space: gymnasium.Space,
        gamma: float = 0.9,
        environment_length: int = 11,
        intensities: np.ndarray | List[int] = [0, 5, 10],
        num_prize_indicators: int = 2,
        initial_value: float = 0.0,
        model_type: str = "perfect",
    ):
        """Initializes an unselective planning agent that equally weights each step in the multi-step return.

        Args:
            action_space (gymnasium.Space): Action space for the environment.
            gamma (float): Discount factor.
            environment_length (int): Size/length of the environment grid.
            intensities (np.ndarray | List[int]): Possible status intensities.
            num_prize_indicators (int): Number of prize indicator bits.
            initial_value (float): Initial Q-value for all state-action pairs.
            model_type (str): The type of dynamics model to use ('perfect', 'expected', or 'sampling').
        """
        super().__init__(
            action_space=action_space,
            gamma=gamma,
            environment_length=environment_length,
            intensities=intensities,
            num_prize_indicators=num_prize_indicators,
            initial_value=initial_value,
        )

        if model_type == "perfect":
            self.dynamics_model = GoRight(
                num_prize_indicators=num_prize_indicators,
                env_length=environment_length,
                has_state_offset=False,
            )
        elif model_type == "expected":
            self.dynamics_model = ExpectationModel(
                num_prize_indicators=num_prize_indicators,
                env_length=environment_length,
                has_state_offset=False,
            )
        elif model_type == "sampling":
            self.dynamics_model = SamplingModel(
                num_prize_indicators=num_prize_indicators,
                env_length=environment_length,
                has_state_offset=False,
            )
        else:
            raise ValueError(
                'Please specify the type of model for unselective planning: "perfect", "expected", or "sampling".'
            )

    def compute_weights(self, td_targets: List[float], **kwargs) -> np.ndarray:
        """Assigns equal weights to each horizon step in the multi-step TD update.

        Args:
            td_targets (List[float]): Computed TD targets for each step.

        Returns:
            np.ndarray: An array of equal weights, summing to 1.
        """
        return np.ones(len(td_targets)) / len(td_targets)
