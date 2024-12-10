"""Module with expectation model."""

from typing import List, Optional

from bbi.models import ExpectationModel


class OneStepPredictedVarianceModel(ExpectationModel):
    metadata = {
        "render_modes": ["human"],
        "environment_name": "1-step Predicted Variance Model",
    }

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = False,
        seed: Optional[int] = None,
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
            has_state_offset=has_state_offset,
            seed=seed,
        )

        pass
