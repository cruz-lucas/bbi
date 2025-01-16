from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class State:
    position: int
    previous_status_indicator: int
    current_status_indicator: int
    prize_indicators: np.ndarray
    offset: np.ndarray | None = None

    mask: List[bool] | None = None

    def __post_init__(self) -> None:
        """By default, this mask drops the 'previous_status_indicator' entry when calling get_observation.

        The order in get_state() is:
          0 -> position
          1 -> previous_status_indicator
          2 -> current_status_indicator
          3.. -> prize_indicators
        We'll mask out index 1 (previous_status_indicator).
        """
        self.mask = np.ones_like(self.get_state(), dtype=bool)
        self.mask[1] = False

    def get_observation(self) -> Dict[str, float | int | np.ndarray]:
        observation = (
            self.get_state() if self.offset is None else self.get_state() + self.offset
        )
        observation = observation[self.mask]

        return dict(
            {
                "position": np.array(observation[0], dtype=np.float32).reshape(
                    1,
                ),
                "status_indicator": np.array(observation[1], dtype=np.float32).reshape(
                    1,
                ),
                "prize_indicators": np.array(observation[2:], dtype=np.float32).reshape(
                    -1,
                ),
            }
        )

    def get_state(self) -> np.ndarray:
        """Get array of state in the order: position, previous status indicator, current status indicator, and a list of prize indicators

        Returns:
            np.ndarray: State
        """
        return np.concatenate(
            [
                [
                    self.position,
                    self.previous_status_indicator,
                    self.current_status_indicator,
                ],
                self.prize_indicators,
            ]
        )

    def _status_to_index(self, status: int) -> int:
        if status == 0:
            return 0
        elif status == 5:
            return 1
        elif status == 10:
            return 2
        else:
            raise ValueError(f"Status {status} not in {{0,5,10}}.")

    def _index_to_status(self, idx: int) -> int:
        if idx == 0:
            return 0
        elif idx == 1:
            return 5
        elif idx == 2:
            return 10
        else:
            raise ValueError(f"Index {idx} not in {{0,1,2}}.")

    def set_state(
        self,
        position: int,
        current_status_indicator: int,
        prize_indicators: List[int] | np.ndarray | int,
        previous_status_indicator: int | None = None,
    ) -> None:
        self.position = position

        if previous_status_indicator not in [0, 5, 10]:
            previous_status_indicator = self._index_to_status(previous_status_indicator)

        if current_status_indicator not in [0, 5, 10]:
            current_status_indicator = self._index_to_status(current_status_indicator)

        self.previous_status_indicator = previous_status_indicator
        self.current_status_indicator = current_status_indicator

        if isinstance(prize_indicators, int):
            for i in range(len(prize_indicators)):
                self.prize_indicators[i] = (prize_indicators >> i) & 1
        else:
            self.prize_indicators = prize_indicators


@dataclass
class Action:
    rigth: int = 1
    left: int = 0


@dataclass
class HistoryEntry:
    state: State
    action: Action
    reward: float
    next_state: State


@dataclass
class Tracker:
    history: List[HistoryEntry] | None = None
    total_reward: int | None = 0
    action_count: int | None = 0

    def record(
        self, state: State, action: Action, reward: float, next_state: State
    ) -> None:
        entry = HistoryEntry(
            state=state, action=action, reward=reward, next_state=next_state
        )
        if self.history is None:
            self.history = [entry]
        else:
            self.history.append(entry)

        self.total_reward += reward
        self.action_count += 1


@dataclass
class BoundingBox:
    state_upper_bound: State
    state_lower_bound: State
    reward_upper_bound: int
    reward_lower_bound: int


@dataclass
class BBITracker:
    history: List[BoundingBox] | None = None

    def record(self, bounding_box: BoundingBox) -> None:
        if self.history is None:
            self.history = [bounding_box]
        else:
            self.history.append(bounding_box)