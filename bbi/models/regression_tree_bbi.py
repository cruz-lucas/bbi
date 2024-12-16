from typing import List, Optional, Tuple

import numpy as np

from bbi.models import BBI


class RegressionTreeNode:
    """A single node in the regression tree."""

    def __init__(self, depth=0, max_depth=5, min_samples_split=10):
        self.is_leaf = False
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.state_mean = None
        self.reward_mean = None
        self.state_bounds = None
        self.reward_bounds = None
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Recursively builds the regression tree.
        X: Features (state, action)
        Y: Targets (next_state, reward)
        """
        # Check for leaf conditions
        if self.depth >= self.max_depth or len(X) < self.min_samples_split:
            self.is_leaf = True
            self.state_mean = np.mean(Y[:, :-1], axis=0)  # next state mean
            self.reward_mean = np.mean(Y[:, -1])  # reward mean
            self.state_bounds = [np.min(Y[:, :-1], axis=0), np.max(Y[:, :-1], axis=0)]
            self.reward_bounds = [np.min(Y[:, -1]), np.max(Y[:, -1])]
            return

        # Find the best split
        best_feature, best_value, best_loss = None, None, float("inf")
        for feature_idx in range(X.shape[1]):
            for split_value in np.unique(X[:, feature_idx]):
                left_indices = X[:, feature_idx] <= split_value
                right_indices = X[:, feature_idx] > split_value

                if (
                    np.sum(left_indices) < self.min_samples_split
                    or np.sum(right_indices) < self.min_samples_split
                ):
                    continue

                left_loss = np.var(Y[left_indices], axis=0).sum()
                right_loss = np.var(Y[right_indices], axis=0).sum()
                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_feature = feature_idx
                    best_value = split_value

        if best_feature is None:
            self.is_leaf = True
            self.state_mean = np.mean(Y[:, :-1], axis=0)
            self.reward_mean = np.mean(Y[:, -1])
            return

        # Perform the split
        self.split_feature = best_feature
        self.split_value = best_value
        left_indices = X[:, best_feature] <= best_value
        right_indices = X[:, best_feature] > best_value

        # Recursively create child nodes
        self.left = RegressionTreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
        )
        self.right = RegressionTreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
        )

        self.left.fit(X[left_indices], Y[left_indices])
        self.right.fit(X[right_indices], Y[right_indices])

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict next state and reward for a single input."""
        if self.is_leaf:
            return self.state_mean, self.reward_mean

        if x[self.split_feature] <= self.split_value:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def get_bounds(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return the bounding-box prediction for a given input."""
        if self.is_leaf:
            return self.state_bounds, self.reward_bounds

        if x[self.split_feature] <= self.split_value:
            return self.left.get_bounds(x)
        else:
            return self.right.get_bounds(x)


class RegressionTreeBBI(BBI):
    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[int] = "human",
        max_depth: int = 5,
        min_samples_split: int = 10,
    ):
        """
        Regression Tree-based model for next-state and reward prediction.
        """
        super().__init__(
            num_prize_indicators=num_prize_indicators,
            env_length=env_length,
            status_intensities=status_intensities,
            has_state_offset=False,
            seed=seed,
        )
        self.tree = RegressionTreeNode(
            max_depth=max_depth, min_samples_split=min_samples_split
        )
        self.state_action_history = []
        self.target_history = []

    def update(
        self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float
    ):
        """Store the experience and periodically retrain the tree."""
        # Feature vector: state + action
        phi = np.concatenate([state, [action]])
        target = np.concatenate([next_state, [reward]])

        self.state_action_history.append(phi)
        self.target_history.append(target)

        # Retrain tree every 500 experiences
        if len(self.state_action_history) % 500 == 0:
            X = np.array(self.state_action_history)
            Y = np.array(self.target_history)
            self.tree.fit(X, Y)

    def step(self, action: int):
        """Perform the step and update the tree."""
        next_state, reward, done, truncated, info = super().step(action)
        self.update(self.state, action, next_state, reward)
        return next_state, reward, done, truncated, info

    def get_next_bounds(
        self, state: np.ndarray, action_bounds: np.ndarray | List[int] = [0, 1]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict bounding-box for next states and rewards given action bounds."""
        state_bounds_list, reward_bounds_list = [], []

        for action in action_bounds:
            phi = np.concatenate([state, [action]])
            state_bounds, reward_bounds = self.tree.get_bounds(phi)
            state_bounds_list.append(state_bounds)
            reward_bounds_list.append(reward_bounds)

        # Combine bounds
        global_state_bounds = np.vstack(
            [
                np.min([b[0] for b in state_bounds_list], axis=0),
                np.max([b[1] for b in state_bounds_list], axis=0),
            ]
        )

        global_reward_bounds = np.array(
            [
                np.min([b[0] for b in reward_bounds_list]),
                np.max([b[1] for b in reward_bounds_list]),
            ]
        )

        return global_state_bounds, global_reward_bounds
