"""Module with BBI model and linear model learning for next-state and reward prediction."""

from typing import List, Optional, Tuple

import numpy as np

from bbi.models import BBI


class LinearBBI(BBI):
    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = "human",
        learning_rate: float = 0.001,
    ):
        """A simple linear model for next-state and reward prediction.

        Args:
            num_prize_indicators (int, optional): _description_. Defaults to 2.
            env_length (int, optional): _description_. Defaults to 11.
            status_intensities (List[int], optional): _description_. Defaults to [0, 5, 10].
            has_state_offset (bool, optional): _description_. Defaults to False.
            seed (Optional[int], optional): _description_. Defaults to None.
            render_mode (Optional[int], optional): _description_. Defaults to "human".
            learning_rate (float, optional): _description_. Defaults to 0.001.
        """
        super().__init__(
            num_prize_indicators=num_prize_indicators,
            env_length=env_length,
            status_intensities=status_intensities,
            has_state_offset=False,
            seed=seed,
        )
        # For linear features: phi(s,a) = [s; a; 1] (bias)
        state_dim = 2 + num_prize_indicators
        action_dim = 2
        self.feature_dim = state_dim + action_dim + 1
        # We model each dimension of the next state and reward:
        # If next_state_dim = state_dim and we have 1 reward dimension,
        # total outputs = state_dim + 1
        # We store weights in a matrix W of shape (feature_dim, state_dim+1)

        # Initialize weights to zero
        self.W = np.zeros((self.feature_dim, state_dim + 1))
        self.lr = learning_rate
        self.state_dim = state_dim
        self.residual_min = np.zeros(state_dim + 1)
        self.residual_max = np.zeros(state_dim + 1)
        self.initialized_residuals = False

    def _feature_vector(self, state: np.ndarray, action: int) -> np.ndarray:
        """_summary_

        Args:
            state (np.ndarray): _description_
            action (int): _description_

        Returns:
            np.ndarray: _description_
        """
        # For discrete actions, we can just append the action as a scalar
        # or one-hot encode if multiple actions. Here we assume a scalar action.
        # phi = [state; action; 1]
        return np.concatenate([state, [action, 1.0]])

    def predict(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """Predict next state and reward given current state and action.

        Args:
            state (np.ndarray): _description_
            action (int): _description_

        Returns:
            Tuple[np.ndarray, float]: _description_
        """
        phi = self._feature_vector(state, action)
        y_pred = phi @ self.W  # shape (state_dim+1,)
        next_state_pred = y_pred[: self.state_dim]
        reward_pred = y_pred[-1]
        return next_state_pred, reward_pred

    def update(
        self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float
    ):
        """Update the linear model using a simple gradient step for least squares:

        loss = 1/2 * ||y - W^T phi||^2
        where y = [next_state; reward].

        Args:
            state (np.ndarray): _description_
            action (int): _description_
            next_state (np.ndarray): _description_
            reward (float): _description_
        """
        phi = self._feature_vector(state, action)  # shape (feature_dim,)
        y = np.concatenate([next_state, [reward]])  # shape (state_dim+1,)

        # Prediction
        y_pred = phi @ self.W  # (state_dim+1,)
        err = y - y_pred

        # Gradient w.r.t W is phi * err (outer product)
        grad = np.outer(phi, err)
        self.W += self.lr * grad

        # Update residual stats for bounding box
        if not self.initialized_residuals:
            self.residual_min = err
            self.residual_max = err
            self.initialized_residuals = True
        else:
            self.residual_min = np.minimum(self.residual_min, err)
            self.residual_max = np.maximum(self.residual_max, err)

    def get_bounds(
        self, state: np.ndarray, action_bounds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For bounding-box inference, we consider the worst-case combination of residuals.

        Since we assume linearity, and no distribution of errors, we just add min/max residuals.

        Args:
            state (np.ndarray): _description_
            action_bounds (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        # For each action in action_bounds:
        # We find min/max predictions by applying residual_min and residual_max.
        pred_list = []
        for a in action_bounds:
            next_state_pred, reward_pred = self.predict(state, a)
            # Construct bounding box by adding min and max residuals
            # NOTE: This is a crude approximation. More nuanced bounding may be needed.
            lower_bound = (
                np.concatenate([next_state_pred, [reward_pred]]) + self.residual_min
            )
            upper_bound = (
                np.concatenate([next_state_pred, [reward_pred]]) + self.residual_max
            )
            pred_list.append((lower_bound, upper_bound))

        # Aggregate all lower bounds and upper bounds
        # lower_bounds: min over all lower_bound predictions
        # upper_bounds: max over all upper_bound predictions
        all_lower = [lb for (lb, _) in pred_list]
        all_upper = [ub for (_, ub) in pred_list]

        global_lower = np.min(all_lower, axis=0)
        global_upper = np.max(all_upper, axis=0)

        # state_bounds and reward_bounds
        # state_dim = self.state_dim
        state_bounds = np.zeros((2, self.state_dim))
        state_bounds[0, :] = global_lower[: self.state_dim]
        state_bounds[1, :] = global_upper[: self.state_dim]

        reward_bounds = np.array([global_lower[-1], global_upper[-1]])
        return state_bounds, reward_bounds

    def step(self, action: int):
        """_summary_

        Args:
            action (int): _description_

        Returns:
            _type_: _description_
        """
        # The step returns the true next state and reward from the environment
        next_state, reward, done, truncated, info = super().step(action)

        # Update the linear model with this transition
        # current_state is self._current_state (from ExpectationModel)
        # next_state is returned by step
        # We assume both are numpy arrays
        self.update(self.state, action, next_state, reward)

        return next_state, reward, done, truncated, info

    def get_next_bounds(
        self, state: np.ndarray, action_bounds: np.ndarray | List[int] = [0, 1]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            state (np.ndarray): _description_
            action_bounds (np.ndarray | List[int], optional): _description_. Defaults to [0, 1].

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        # Use the linear model to get bounding boxes
        self.state_bounding_box, self.reward_bounding_box = self.get_bounds(
            state, action_bounds
        )
        return self.state_bounding_box, self.reward_bounding_box
