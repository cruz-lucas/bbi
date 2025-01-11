from typing import Tuple

from bbi.agents import (
    BaseQAgent,
    QLearningAgent,
    SelectivePlanningAgent,
    UnselectivePlanningAgent,
)
from bbi.environments import BaseEnv, GoRight
from bbi.models import ExpectationModel, SamplingModel


def select_agent_and_model(
    model_id: str,
    discount: float,
    num_prize_indicators: int,
    environment_length: int,
    number_intensities: int,
    tau: float | None = None,
) -> Tuple[BaseQAgent, BaseEnv | None]:
    if model_id == "perfect":
        model = GoRight(
            num_prize_indicators=num_prize_indicators,
            env_length=environment_length,
            has_state_offset=False,
        )

        agent = UnselectivePlanningAgent(
            discount=discount,
            number_positions=environment_length,
            number_intensities=number_intensities,
            number_prize_indicators=num_prize_indicators,
        )

    elif model_id == "expected":
        model = ExpectationModel(
            num_prize_indicators=num_prize_indicators,
            env_length=environment_length,
            has_state_offset=False,
        )

        agent = UnselectivePlanningAgent(
            discount=discount,
            number_positions=environment_length,
            number_intensities=number_intensities,
            number_prize_indicators=num_prize_indicators,
        )

    elif model_id == "sampling":
        model = SamplingModel(
            num_prize_indicators=num_prize_indicators,
            env_length=environment_length,
            has_state_offset=False,
        )

        agent = UnselectivePlanningAgent(
            discount=discount,
            number_positions=environment_length,
            number_intensities=number_intensities,
            number_prize_indicators=num_prize_indicators,
        )

    elif model_id == "bbi":
        model = ExpectationModel(
            num_prize_indicators=num_prize_indicators,
            env_length=environment_length,
            has_state_offset=False,
        )

        agent = SelectivePlanningAgent(
            discount=discount,
            number_positions=environment_length,
            number_intensities=number_intensities,
            number_prize_indicators=num_prize_indicators,
            tau=tau,
        )

    else:
        model = None
        agent = QLearningAgent(
            discount=discount,
            number_positions=environment_length,
            number_intensities=number_intensities,
            number_prize_indicators=num_prize_indicators,
        )

    return agent, model
