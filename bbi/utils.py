"""Utility funcitons."""

import argparse
from typing import Any, Dict

import pygame
import yaml
from pygame.locals import K_ESCAPE, K_LEFT, K_RIGHT, KEYDOWN, K_m, K_r

# Import your environments and models
from bbi.environments import GoRight
from bbi.models import BBI, ExpectationModel, SamplingModel

# List of model classes to cycle through
MODEL_CLASSES = [GoRight, SamplingModel, ExpectationModel, BBI]


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


def _status_color(value: int, is_text: bool = False) -> str:
    if not is_text:
        return {0: "#1B1D1F", 5: "#454C53", 10: "#C9CDD2"}.get(value, "#C9CDD2")
    return {0: "#F7F8F9", 5: "#F7F8F9", 10: "#26282B"}.get(value, "#F7F8F9")


def _draw_action_arrow(env, grid_left: int) -> None:
    arr_x = grid_left + int(env.last_pos) * env.cell_size + env.cell_size // 2
    arr_y = env.grid_top + env.cell_size // 2
    color = (255, 0, 0)
    if env.last_action == 0:  # Left action
        pygame.draw.polygon(
            env.screen,
            color,
            [
                (arr_x - env.cell_size // 2, arr_y),
                (arr_x - env.cell_size // 2 + 10, arr_y + 10),
                (arr_x - env.cell_size // 2 + 10, arr_y - 10),
            ],
        )
    else:  # Right action
        pygame.draw.polygon(
            env.screen,
            color,
            [
                (arr_x + env.cell_size // 2, arr_y),
                (arr_x + env.cell_size // 2 - 10, arr_y + 10),
                (arr_x + env.cell_size // 2 - 10, arr_y - 10),
            ],
        )


def _draw_status_indicators(
    env, env_name: str, prev_status: int, current_status: int
) -> None:
    box_size = 40
    status_x = env.margin
    status_y = 20
    font = env.font
    screen = env.screen

    if env_name == "GoRight":
        # Prev
        pygame.draw.rect(
            screen, _status_color(prev_status), (status_x, status_y, box_size, box_size)
        )
        prev_text = font.render(
            str(prev_status), True, _status_color(prev_status, True)
        )
        screen.blit(
            prev_text,
            (
                status_x + (box_size - prev_text.get_width()) / 2,
                status_y + (box_size - prev_text.get_height()) / 2,
            ),
        )
        prev_label = font.render("Prev", True, (0, 0, 0))
        screen.blit(prev_label, (status_x, status_y + box_size + 5))

        # Curr
        pygame.draw.rect(
            screen,
            _status_color(current_status),
            (status_x + 50, status_y, box_size, box_size),
        )
        curr_text = font.render(
            str(current_status), True, _status_color(current_status, True)
        )
        screen.blit(
            curr_text,
            (
                status_x + 50 + (box_size - curr_text.get_width()) / 2,
                status_y + (box_size - curr_text.get_height()) / 2,
            ),
        )
        curr_label = font.render("Curr", True, (0, 0, 0))
        screen.blit(curr_label, (status_x + 50, status_y + box_size + 5))
    else:
        # Only Curr
        pygame.draw.rect(
            screen,
            _status_color(current_status),
            (status_x, status_y, box_size, box_size),
        )
        curr_text = font.render(
            str(current_status), True, _status_color(current_status, True)
        )
        screen.blit(
            curr_text,
            (
                status_x + (box_size - curr_text.get_width()) / 2,
                status_y + (box_size - curr_text.get_height()) / 2,
            ),
        )
        curr_label = font.render("Curr", True, (0, 0, 0))
        screen.blit(curr_label, (status_x, status_y + box_size + 5))


def _draw_prize_indicators(env, width: int, prize_indicators) -> None:
    lamps_x = width - env.margin - env.num_prize_indicators * 40
    lamp_y = 30
    for i, val in enumerate(prize_indicators):
        lamp_x = lamps_x + i * 40
        img = env.lamp_on if val > 0.5 else env.lamp_off
        env.screen.blit(img, (lamp_x, lamp_y))


def _draw_info_text(env) -> None:
    font = env.font
    screen = env.screen
    margin = env.margin

    reward_label = font.render(f"Last Reward: {env.last_reward}", True, (0, 0, 0))
    screen.blit(reward_label, (margin, 170))

    cum_reward_label = font.render(
        f"Cumulative Reward: {env.total_reward}", True, (0, 0, 0)
    )
    screen.blit(cum_reward_label, (margin, 190))

    actions_label = font.render(f"Actions taken: {env.action_count}", True, (0, 0, 0))
    screen.blit(actions_label, (margin, 210))


def _draw_bounding_boxes(env) -> None:
    font = env.font
    screen = env.screen
    margin = env.margin
    bbox_y_start = 250

    if hasattr(env, "state_bounding_box") and env.state_bounding_box is not None:
        state_min = env.state_bounding_box[0]
        state_max = env.state_bounding_box[1]
        state_min_str = "State Lower Bound: " + str(state_min)
        state_max_str = "State Upper Bound: " + str(state_max)
        min_text = font.render(state_min_str, True, (0, 0, 0))
        max_text = font.render(state_max_str, True, (0, 0, 0))
        screen.blit(min_text, (margin, bbox_y_start + 5))
        screen.blit(max_text, (margin, bbox_y_start + 25))

    if hasattr(env, "reward_bounding_box") and env.reward_bounding_box is not None:
        reward_min, reward_max = env.reward_bounding_box
        reward_str = f"Reward Bounds: [{reward_min}, {reward_max}]"
        reward_text = font.render(reward_str, True, (0, 0, 0))
        screen.blit(reward_text, (margin, bbox_y_start + 45))


def _init_pygame(env, width, height):
    env_name = env.metadata.get("environment_name")

    pygame.init()
    env.screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"{env_name or 'Environment'}")
    env.clock = pygame.time.Clock()
    env.font = pygame.font.SysFont("Arial", 20)
    env.lamp_on = pygame.image.load(
        "bbi/environments/resources/lamp_on.png"
    ).convert_alpha()
    env.lamp_off = pygame.image.load(
        "bbi/environments/resources/lamp_off.png"
    ).convert_alpha()
    env.robot = pygame.image.load(
        "bbi/environments/resources/robot.png"
    ).convert_alpha()


def render_env(env, model_classes, current_model_index):
    env_name = env.metadata.get("environment_name")
    width = env.margin * 2 + env.cell_size * env.length
    height = env.top_area_height + env.margin + env.cell_size + env.bottom_area_height

    # Initialize if needed
    if not hasattr(env, "screen") or env.screen is None:
        _init_pygame(env, width, height)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                raise SystemExit
            elif event.key == K_LEFT:
                if env_name == "BBI":
                    env.get_next_bounds(env.state)
                env.step(0)
            elif event.key == K_RIGHT:
                if env_name == "BBI":
                    env.get_next_bounds(env.state)
                env.step(1)
            elif event.key == K_m:
                # Cycle models
                current_model_index = (current_model_index + 1) % len(model_classes)
                env = model_classes[current_model_index](render_mode="human")
                env.reset()
                height = (
                    env.top_area_height
                    + env.margin
                    + env.cell_size
                    + env.bottom_area_height
                )
                _init_pygame(env, width, height)
            elif event.key == K_r:
                # Reset current environment
                env.reset()

    env.screen.fill((255, 255, 255))

    # Extract state info
    position = int(env.state[0])
    current_status = int(env.state[1])
    prize_indicators = env.state[2:]
    prev_status = int(env.previous_status) if env.previous_status is not None else 0

    # Draw grid
    grid_left = env.margin
    for i in range(env.length):
        rect_x = grid_left + i * env.cell_size
        rect_y = env.grid_top
        pygame.draw.rect(
            env.screen, (0, 0, 0), (rect_x, rect_y, env.cell_size, env.cell_size), 2
        )

    # Draw agent
    agent_x = grid_left + position * env.cell_size + env.cell_size // 4
    agent_y = env.grid_top + env.cell_size // 4
    env.screen.blit(env.robot, (agent_x, agent_y))

    # Draw last action arrow if any
    if env.last_action is not None and env.last_pos is not None:
        _draw_action_arrow(env, grid_left)

    _draw_status_indicators(env, env_name, prev_status, current_status)
    _draw_prize_indicators(env, width, prize_indicators)
    _draw_info_text(env)

    if env_name == "BBI":
        _draw_bounding_boxes(env)

    pygame.display.flip()
    env.clock.tick(10)

    return env, current_model_index


if __name__ == "__main__":
    # Initialize the first environment
    current_model_index = 0
    env = MODEL_CLASSES[current_model_index](render_mode="human")
    env.reset()

    running = True
    while running:
        # Update env and current_model_index on each render call
        env, current_model_index = render_env(env, MODEL_CLASSES, current_model_index)
        # The loop continues until ESC or window close is triggered
