"""Utility funcitons."""

import argparse
from typing import Any, Dict

import yaml


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Q-Learning Agent with Hyperparameters"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bbi/config/config_q_learning.yaml",
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
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for Q-learning"
    )
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


def render_simulator():
    import pygame
    from pygame.locals import K_ESCAPE, K_LEFT, K_RIGHT, KEYDOWN, K_m, K_r

    from bbi.environments import GoRight
    from bbi.models import BBI, ExpectationModel, SamplingModel

    # List of models to cycle through
    MODEL_CLASSES = [GoRight, SamplingModel, ExpectationModel, BBI]
    # MODEL_NAMES = ["GoRight", "SamplingModel", "ExpectationModel", "BBI"]
    current_model_index = 0
    env = MODEL_CLASSES[current_model_index]()
    obs, info = env.reset()

    pygame.init()
    pygame.font.SysFont("Arial", 20)
    running = True

    while running:
        env.render()

        screen = pygame.display.get_surface()
        if screen is None:
            # If env.render doesn't set up a display, do it here
            screen = pygame.display.set_mode((800, 600))

        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                break
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                    pygame.quit()
                    break
                elif event.key == K_LEFT:
                    env.step(0)
                elif event.key == K_RIGHT:
                    env.step(1)
                elif event.key == K_m:
                    # Change model
                    current_model_index = (current_model_index + 1) % len(MODEL_CLASSES)
                    env = MODEL_CLASSES[current_model_index]()
                    env.reset()
                elif event.key == K_r:
                    # Reset model
                    env.reset()

        # pygame.time.delay(100)


if __name__ == "__main__":
    render_simulator()
