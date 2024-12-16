from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bbi.models import BBI


class NeuralBBI(BBI):
    """BBI model using a neural network for next-state and reward prediction with bounding-box estimation."""

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[int] = "human",
        learning_rate: float = 0.001,
        hidden_units: int = 128,
    ):
        super().__init__(
            num_prize_indicators=num_prize_indicators,
            env_length=env_length,
            status_intensities=status_intensities,
            has_state_offset=has_state_offset,
            seed=seed,
        )
        self.learning_rate = learning_rate
        self.state_dim = 2 + num_prize_indicators
        self.action_dim = 1  # Assume scalar action
        self.output_dim = self.state_dim + 1  # Next-state + reward prediction

        # Define neural network with three outputs: mean, upper, and lower quantiles
        self.model = QuantileNN(
            input_dim=self.state_dim + self.action_dim,
            output_dim=self.output_dim,
            hidden_units=hidden_units,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = PinballLoss()

        self.buffer = []  # Experience replay buffer for training
        self.batch_size = 32

    def _feature_vector(self, state: np.ndarray, action: int) -> torch.Tensor:
        """Convert state-action pair to a feature vector."""
        return torch.tensor(np.concatenate([state, [action]]), dtype=torch.float32)

    def predict(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """
        Predict next state and reward, using the mean prediction.
        """
        self.model.eval()
        with torch.no_grad():
            phi = self._feature_vector(state, action).unsqueeze(
                0
            )  # Add batch dimension
            mean_prediction, _, _ = self.model(phi)
        mean_prediction = mean_prediction.squeeze(0).numpy()
        next_state_pred = mean_prediction[: self.state_dim]
        reward_pred = mean_prediction[-1]
        return next_state_pred, reward_pred

    def update(
        self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float
    ):
        """Store experience and train the model."""
        # Add experience to the buffer
        self.buffer.append((state, action, next_state, reward))
        if len(self.buffer) >= self.batch_size:
            self._train()

    def _train(self):
        """Train the neural network using experiences from the buffer."""
        self.model.train()

        # Sample a minibatch
        batch = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, next_states, rewards = zip(*[self.buffer[i] for i in batch])

        # Convert to tensors
        inputs = torch.stack(
            [self._feature_vector(s, a) for s, a in zip(states, actions)]
        )
        targets = torch.tensor(
            np.hstack([next_states, np.array(rewards).reshape(-1, 1)]),
            dtype=torch.float32,
        )

        # Forward pass
        mean_pred, lower_pred, upper_pred = self.model(inputs)

        # Compute losses using Pinball loss for quantiles
        loss = self.loss_fn(mean_pred, lower_pred, upper_pred, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_bounds(
        self, state: np.ndarray, action_bounds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bounding-box estimates for next state and reward."""
        lower_bounds = []
        upper_bounds = []

        for action in action_bounds:
            with torch.no_grad():
                phi = self._feature_vector(state, action).unsqueeze(
                    0
                )  # Add batch dimension
                _, lower_pred, upper_pred = self.model(phi)

            lower_pred = lower_pred.squeeze(0).numpy()
            upper_pred = upper_pred.squeeze(0).numpy()

            lower_bounds.append(lower_pred)
            upper_bounds.append(upper_pred)

        global_lower = np.min(lower_bounds, axis=0)
        global_upper = np.max(upper_bounds, axis=0)

        state_bounds = np.zeros((2, self.state_dim))
        state_bounds[0, :] = global_lower[: self.state_dim]
        state_bounds[1, :] = global_upper[: self.state_dim]

        reward_bounds = np.array([global_lower[-1], global_upper[-1]])
        return state_bounds, reward_bounds


class QuantileNN(nn.Module):
    """Neural network predicting mean, lower, and upper quantiles."""

    def __init__(self, input_dim: int, output_dim: int, hidden_units: int = 128):
        super(QuantileNN, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_units, output_dim)
        self.lower_head = nn.Linear(hidden_units, output_dim)
        self.upper_head = nn.Linear(hidden_units, output_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.shared(x)
        mean = self.mean_head(x)
        lower = self.lower_head(x)
        upper = self.upper_head(x)
        return mean, lower, upper


class PinballLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantiles=(0.05, 0.95)):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(
        self,
        mean_pred: torch.Tensor,
        lower_pred: torch.Tensor,
        upper_pred: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Pinball loss for lower quantile
        lower_loss = torch.maximum(
            self.quantiles[0] * (targets - lower_pred),
            (1 - self.quantiles[0]) * (lower_pred - targets),
        )
        # Pinball loss for upper quantile
        upper_loss = torch.maximum(
            self.quantiles[1] * (targets - upper_pred),
            (1 - self.quantiles[1]) * (upper_pred - targets),
        )
        # Mean squared error for mean prediction
        mean_loss = torch.nn.functional.mse_loss(mean_pred, targets)

        return mean_loss + lower_loss.mean() + upper_loss.mean()
