from itertools import product
from typing import Dict, Optional, Tuple, Union

import numpy as np

from bbi.models.model_base import ModelBase, ObsType


def softmin(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Compute a softmin distribution over an array x using temperature tau.

    Args:
        x: Array of values.
        tau: Temperature parameter (default: 1.0).

    Returns:
        A probability distribution over x computed via softmin.
    """
    scaled = np.exp(-np.array(x) / tau)
    return scaled / np.sum(scaled)


class BoundingBoxPlanningAgent:
    """A planning agent that implements a bounding-box inference method for selective planning.

    This agent uses tabular Q-learning augmented with model-based value expansion.
    In each update, it first computes a 1–step TD target from the real transition.
    Then, for additional planning steps (up to a given horizon), it queries a predictive
    model that returns not only an expected next state and reward but also lower and upper
    bounds for both. For each rollout step, the agent computes a target using the expected
    prediction and also derives an uncertainty measure (the difference between the upper and
    lower targets). These targets are then weighted via a softmin function over uncertainties
    and combined to update Q-values.
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        step_size: float,
        discount_rate: float,
        epsilon: float,
        horizon: int,
        initial_value: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            state_shape: Shape of the state space (for storing Q in a numpy array).
            num_actions: Number of possible actions.
            step_size: Learning rate for Q–updates.
            discount_rate: Discount factor (gamma).
            epsilon: Epsilon for epsilon–greedy exploration.
            horizon: Maximum planning horizon H.
            initial_value: Value to initialize the q-values.
            seed: Optional random seed.
        """
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.step_size = step_size
        self.discount_rate = discount_rate
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

        self.Q = np.full(state_shape + (num_actions,), fill_value=initial_value)

        self.td_errors = []

    def act(self, obs: ObsType) -> int:
        """Select an action using an epsilon–greedy strategy based on Q–values.

        Args:
            state: Current state as a discrete tuple.

        Returns:
            The chosen action as an integer.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        else:
            state = self._round_obs(obs)
            q_values = self.Q[state]
            return int(self.rng.choice(np.flatnonzero(q_values == q_values.max())))

    def update(
        self,
        obs: Dict[str, Union[np.float32, np.ndarray]],
        action: int,
        next_obs: Dict[str, Union[np.float32, np.ndarray]],
        reward: float,
        model: ModelBase,
    ) -> None:
        """Perform a bounding-box inference update of Q(s, a).

        The update procedure is as follows:
          1. Compute the 1–step TD target from the observed (real) transition.
          2. For each planning step h = 1 ... H–1:
             a. Use the predictive model to obtain an imaginary transition from the current planning
                state. The model returns:
                - an expected next state and reward,
                - a lower bound state and reward,
                - an upper bound state and reward.
             b. Compute:
                - the expected target: r_pred + γ * maxₐ Q(s_pred, a),
                - lower target:    r_lower + γ * (minₛ∈box maxₐ Q(s, a))
                - upper target:    r_upper + γ * (maxₛ∈box maxₐ Q(s, a))
             c. Define the uncertainty as (upper target – lower target) and record the expected target.
          3. Combine the targets from all steps using a softmin weighting over uncertainties.
          4. Update Q(s, a) with a gradient step toward the weighted target.

        Args:
            obs: Current observation (a dict with keys "position", "status_indicator", and "prize_indicators").
            action: Action taken in the current state.
            next_obs: Observed next observation.
            reward: Observed reward after taking the action.
            model: A predictive model providing a method
                   predict(state: Tuple[int, ...], action: int) ->
                     (expected_state, expected_reward,
                      lower_state, lower_reward,
                      upper_state, upper_reward).

        Returns:
            None.
        """
        gamma = self.discount_rate

        s = self._round_obs(obs)
        s_prime = self._round_obs(next_obs)

        target_real = reward + gamma * np.max(self.Q[s_prime])
        targets = [target_real]
        uncertainties = [0.0]

        planning_state = s_prime
        planning_action = int(np.argmax(self.Q[s_prime]))

        state_bb = (planning_state, planning_state)
        action_bb = (planning_action, planning_action)

        for h in range(1, self.horizon):
            prediction = model.predict(
                obs=planning_state,
                action=planning_action,
                state_bb=state_bb,
                action_bb=action_bb,
            )

            if len(prediction) == 6:  # use bounding box
                (
                    pred_state,
                    pred_reward,
                    low_state,
                    low_reward,
                    up_state,
                    up_reward,
                ) = prediction
                state_bb = (low_state, up_state)

                q_pred = np.max(self.Q[pred_state])
                target_pred = pred_reward + gamma * q_pred

                q_low, a_low, q_up, a_up = self.bounding_box_propagation(
                    low_state, up_state
                )
                action_bb = (a_low, a_up)
                target_low = low_reward + gamma * q_low
                target_up = up_reward + gamma * q_up
                uncertainty = target_up - target_low

                targets.append(target_pred)
                uncertainties.append(uncertainty)

                planning_state = pred_state
                planning_action = int(np.argmax(self.Q[planning_state]))

            else:
                (pred_state, pred_reward) = prediction

                q_pred = np.max(self.Q[pred_state])
                target_pred = pred_reward + gamma * q_pred
                targets.append(target_pred)
                uncertainties.append(0)

                planning_state = pred_state
                planning_action = int(np.argmax(self.Q[planning_state]))

        weights = softmin(np.array(uncertainties))
        final_target = np.dot(weights, np.array(targets))

        td_error = final_target - self.Q[s][action]

        self.Q[s][action] += self.step_size * td_error

        self.td_errors.append(td_error)

        return td_error

    def bounding_box_propagation(
        self, low_state: Tuple[int, ...], up_state: Tuple[int, ...]
    ) -> Tuple[float, float]:
        """Compute the minimum and maximum greedy Q-values over all states in the bounding box.

        A state s lies in the bounding box defined by low_state and up_state if and only if
          low_state[i] <= s[i] <= up_state[i] for all dimensions i.

        Args:
            low_state: Lower bound state as a tuple.
            up_state: Upper bound state as a tuple.

        Returns:
            A tuple (q_low, q_up) where q_low is the minimum and q_up is the maximum
            greedy Q-value among all states in the bounding box.
        """
        dims = len(low_state)
        low_candidates = []
        up_candidates = []
        ranges = [range(low_state[i], up_state[i] + 1) for i in range(dims)]
        for s in product(*ranges):
            q_vals = self.Q[s]
            max_q = np.max(q_vals)
            best_actions = np.flatnonzero(q_vals == max_q)
            chosen_action = int(self.rng.choice(best_actions))
            low_candidates.append((max_q, chosen_action))
            up_candidates.append((max_q, chosen_action))

        q_low = min(low_candidates, key=lambda x: x[0])[0]
        low_actions = [a for (q, a) in low_candidates if q == q_low]
        a_low = int(self.rng.choice(low_actions))

        q_up = max(up_candidates, key=lambda x: x[0])[0]
        up_actions = [a for (q, a) in up_candidates if q == q_up]
        a_up = int(self.rng.choice(up_actions))

        if a_up < a_low:
            # just reordering actions if a_low < a_up
            return (q_low, a_up, q_up, a_low)
        return (q_low, a_low, q_up, a_up)

    def _round_obs(self, obs: ObsType) -> Tuple[int, int, np.ndarray]:
        """Convert a continuous observation into a discrete (tabular) state.

        The conversion is performed as follows:
          - The "position" is rounded to the nearest integer.
          - The "status_indicator" is discretized into three bins:
                <= 2.5 -> 0,
                <= 7.5 -> 1,
                > 7.5  -> 2.
          - Each element in "prize_indicators" is thresholded at 0.5.

        Args:
            obs: A dictionary with keys "position", "status_indicator", and "prize_indicators".

        Returns:
            A tuple representing the discrete state.
        """
        pos = int(np.round(obs["position"]).item())
        status_val = float(obs["status_indicator"].item())
        if status_val <= 2.5:
            intensity_value = 0
        elif status_val <= 7.5:
            intensity_value = 1
        else:
            intensity_value = 2

        prize_indicators = [
            int(obs["prize_indicators"][i] >= 0.5)
            for i in range(len(obs["prize_indicators"]))
        ]
        return (pos, intensity_value, *prize_indicators)
