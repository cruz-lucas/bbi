"""Utility funcitons."""
import argparse
from typing import Any, Dict

import yaml


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train Q-Learning Agent with Hyperparameters")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=100,
        help="Number of training steps to log training progress",
    )
    parser.add_argument("--training_loops", type=int, help="Number of training loops")
    parser.add_argument("--n_steps", type=int, help="Number of steps per episode")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for Q-learning")
    parser.add_argument("--discount_factor", type=float, help="Discount factor (gamma)")
    parser.add_argument("--n_seeds", type=int, help="Number of seeds")
    parser.add_argument("--start_seed", type=int, help="Start seed")

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads configuration from a YAML file and updates it with command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments if provided
    if args.training_loops is not None:
        config["training"]["training_loops"] = args.training_loops
    if args.n_steps is not None:
        config["training"]["n_steps"] = args.n_steps
    if args.learning_rate is not None:
        config["agent"]["learning_rate"] = args.learning_rate
    if args.discount_factor is not None:
        config["agent"]["gamma"] = args.discount_factor
    if args.n_seeds is not None:
        config["training"]["n_seeds"] = args.n_seeds
    if args.start_seed is not None:
        config["training"]["start_seed"] = args.start_seed

    config["verbose"] = args.verbose
    return config
