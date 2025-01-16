"""_summary_"""

from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bbi.environments.base_env import BaseEnv
from bbi.utils.constants import STATUS_TRANSITION
from bbi.utils.dataclasses import Action, State, Tracker


class GoRight(BaseEnv):
    """Custom Gymnasium environment for the "Go Right" task.

    The agent moves along a 1D grid, aiming to collect prizes at the end.
    """

    metadata = {"render_modes": ["human"], "environment_name": "GoRight"}

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the GoRight environment.

        Args:
            num_prize_indicators: Number of prize indicators.
            env_length: Length of the grid.
            status_intensities: Possible status intensities.
            has_state_offset: Whether to add noise to observations.
            seed: Seed for reproducibility.
        """
        super().__init__()
        self.num_prize_indicators = num_prize_indicators
        self.length = env_length
        self.max_intensity = max(status_intensities)
        self.intensities = status_intensities
        self.has_state_offset = has_state_offset

        self.max_offset_pos = 0.25
        self.max_status_pos = 1.25
        self.max_prize_pos = 0.25

        self.action_space: gym.Space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=-0.25,
                    high=11.25,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "status_indicator": spaces.Box(
                    low=-1.25,
                    high=11.25,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "prize_indicators": spaces.Box(
                    low=-0.25,
                    high=1.25,
                    shape=(self.num_prize_indicators,),
                    dtype=np.float32,
                ),
            }
        )

        self.state: Optional[State] = None
        self.tracker: Tracker = Tracker()

        self.seed(seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[State, Dict[str, Any]]:
        """Resets the environment to its initial state.

        Args:
            seed: Seed for reproducibility.
            options: Additional options.

        Returns:
            observation: The initial observation of the environment.
            info: Additional info dictionary.
        """
        super().reset(seed=seed)
        self.tracker: Tracker = Tracker()

        position_offset = self.np_random.uniform(-0.25, 0.25)
        status_offset = self.np_random.uniform(-1.25, 1.25)
        prize_offset = self.np_random.uniform(
            -0.25, 0.25, size=self.num_prize_indicators
        )
        self.offset = (
            np.concatenate(
                [
                    [position_offset, status_offset, status_offset],
                    prize_offset,
                ]
            )
            if self.has_state_offset
            else None
        )

        self.state = State(
            position=0,
            previous_status_indicator=self.np_random.choice(self.intensities),
            current_status_indicator=self.np_random.choice(self.intensities),
            prize_indicators=np.zeros((self.num_prize_indicators)),
            offset=self.offset,
        )

        return self.state.get_observation(), {}

    def step(self, action: int) -> Tuple[State, float, bool, bool, Dict[str, Any]]:
        """Executes one time step within the environment.

        Args:
            action: The action taken by the agent (0: left, 1: right).

        Returns:
            observation: The next observation.
            reward: The reward obtained from taking this step.
            terminated: Whether the episode has ended.
            truncated: Whether the episode was truncated.
            info: Additional info dictionary.
        """
        if self.state is None:
            raise ValueError("State has not been initialized.")

        current_state: State = deepcopy(self.state)

        next_pos = self._compute_next_position(action)
        next_status = STATUS_TRANSITION.get(
            (
                current_state.previous_status_indicator,
                current_state.current_status_indicator,
            ),
            None,
        )

        next_prize_indicators = self._compute_next_prize_indicators(
            next_pos, next_status
        )

        reward = self._compute_reward(next_prize_indicators, action)

        self.state.set_state(
            position=next_pos,
            previous_status_indicator=current_state.current_status_indicator,
            current_status_indicator=next_status,
            prize_indicators=next_prize_indicators,
        )

        self.tracker.record(
            state=current_state, action=action, reward=reward, next_state=self.state
        )

        return self.state.get_observation(), reward, False, False, {}

    def _compute_next_position(self, action: int, position: int | None = None,) -> int:
        """Calculates the next position based on the current position and action.

        Args:
            action (int): The action taken by the agent (0: left, 1: right).
            state (State): The current state.

        Returns:
            int: The next position
        """
        position = self.state.position if position is None else position
        direction = 1 if action > 0 else -1
        return np.clip(position + direction, 0, self.length - 1)

    def _compute_next_prize_indicators(
        self,
        next_position: float,
        next_status: int,
        prize_indicators: np.ndarray | None = None,
        position: int | None = None,
    ) -> np.ndarray:
        """Computes the next prize indicators based on the current state.

        Args:
            next_position (float): The agent's next position in the grid.
            next_status (int): The next status intensity of the environment after the move.
            current_state (State):

        Returns:
            np.ndarray: An updated array of prize indicators.
        """
        prize_indicators = np.array(self.state.prize_indicators) if prize_indicators is None else np.array(prize_indicators)
        pos = self.state.position if position is None else position

        if int(next_position) == self.length - 1:
            if int(pos) == self.length - 2:
                if next_status == self.max_intensity:
                    return np.ones_like(prize_indicators, dtype=int)
            elif all(prize_indicators == 1):
                return prize_indicators
            else:
                return self._shift_prize_indicators(prize_indicators)
        return np.zeros_like(prize_indicators, dtype=int)

    def _shift_prize_indicators(self, prize_indicators: np.ndarray) -> np.ndarray:
        """Shift prize indicators forward, simulating their movement.

        Args:
            prize_indicators (np.ndarray): Current prize indicators.

        Returns:
            np.ndarray: Updated prize indicators after shifting.
        """
        if all(prize_indicators < 0.5):
            prize_indicators[0] = 1
            prize_indicators[1:] = 0
        else:
            one_index = np.argmax(prize_indicators)
            prize_indicators[one_index] = 0
            if one_index < self.num_prize_indicators - 1:
                prize_indicators[one_index + 1] = 1
        return prize_indicators

    def _compute_reward(
        self, next_prize_indicators: np.ndarray, action: int
    ) -> float:
        """Compute the reward based on action and prize indicators.

        Args:
            next_prize_indicators (np.ndarray): Updated prize indicators.
            action (int): Action taken by the agent.
            state (State):

        Returns:
            float: Calculated reward.
        """
        if all(next_prize_indicators == 1) and int(self.state.position) == self.length - 1:
            return 3.0
        return 0.0 if action == Action.left else -1.0
