import numpy as np


def compute_epsilon_greedy_action_probs(
    q_vals: np.ndarray, epsilon: float
) -> np.ndarray:
    """Takes in Q-values and produces epsilon-greedy action probabilities

    where ties are broken evenly.

    Args:
        q_vals: a numpy array of action values
        epsilon: epsilon-greedy epsilon in ([0,1])

    Returns:
        numpy array of action probabilities
    """
    assert len(q_vals.shape) == 1

    uniform_probabilities = np.ones_like(q_vals) / len(q_vals)

    ties = np.argwhere(q_vals == q_vals.max()).flatten()

    greedy_probabilities = np.zeros_like(q_vals)
    greedy_probabilities[ties] = 1.0 / len(ties)

    action_probabilities = (
        epsilon * uniform_probabilities + (1 - epsilon) * greedy_probabilities
    )

    assert np.isclose(action_probabilities.sum(), 1.0, atol=1e-6)

    assert action_probabilities.shape == q_vals.shape
    return action_probabilities
