from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pygame.locals import K_LEFT, K_RIGHT, KEYDOWN

STATUS_TABLE = {
    (0, 0): 5,
    (0, 5): 0,
    (0, 10): 5,
    (5, 0): 10,
    (5, 5): 10,
    (5, 10): 10,
    (10, 0): 0,
    (10, 5): 5,
    (10, 10): 0,
}


class GoRight(gym.Env):
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
        self.previous_status: Optional[int] = None
        self.has_state_offset = has_state_offset

        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Box(
            low=0.0,
            high=max(env_length - 1, self.max_intensity, 1.0),
            shape=(2 + num_prize_indicators,),
            dtype=np.float32,
        )

        self.state: Optional[np.ndarray] = None
        self.previous_status: Optional[int] = None
        self.render_mode = "human"
        self.last_action = None
        self.last_pos = None
        self.last_reward = 0.0

        # Track cumulative reward and action count per episode
        self.total_reward = 0.0
        self.action_count = 0

        self.seed(seed)

        # Rendering related attributes
        self.screen = None
        self.clock = None
        self.cell_size = 60
        self.margin = 50
        self.font = None

        # Image sprites for lamps
        self.lamp_on = None
        self.lamp_off = None
        self.robot = None

        # Reset button attributes
        self.reset_btn_width = 80
        self.reset_btn_height = 30

    def seed(self, seed: Optional[int] = None) -> None:
        """Sets the seed for the environment's random number generator.

        Args:
            seed: The seed value.
        """
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to its initial state.

        Args:
            seed: Seed for reproducibility.
            options: Additional options.

        Returns:
            observation: The initial observation of the environment.
            info: Additional info dictionary.
        """
        super().reset(seed=seed)
        self.state = np.zeros(2 + self.num_prize_indicators, dtype=np.float32)
        self.state[1] = self.np_random.choice(self.intensities)
        self.previous_status = self.np_random.choice(self.intensities)
        self.last_action = None
        self.last_pos = None
        self.last_reward = 0.0

        # Reset cumulative stats
        self.total_reward = 0.0
        self.action_count = 0

        if self.has_state_offset:
            self.position_offset = self.np_random.uniform(-0.25, 0.25)
            self.status_indicator_offset = self.np_random.uniform(-1.25, 1.25)
            self.prize_indicator_offsets = self.np_random.uniform(
                -0.25, 0.25, size=self.num_prize_indicators
            )

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
        position, current_status, *prize_indicators = self.state

        direction = 1 if action > 0 else -1
        next_pos = np.clip(position + direction, 0, self.length - 1)

        next_status = STATUS_TABLE.get(
            (self.previous_status, current_status), current_status
        )
        next_prize_indicators = self._compute_next_prize_indicators(
            next_pos, position, next_status, np.array(prize_indicators)
        )

        # Update state
        self.state[0] = next_pos
        self.state[1] = next_status
        self.state[2:] = next_prize_indicators

        reward = self._compute_reward(next_prize_indicators, action, position)
        self.previous_status = current_status
        self.last_action = action
        self.last_pos = position
        self.last_reward = reward

        # Update cumulative stats
        self.total_reward += reward
        self.action_count += 1

        return self._get_observation(), reward, False, False, {}

    def _compute_next_prize_indicators(
        self,
        next_position: float,
        position: float,
        next_status: int,
        prize_indicators: np.ndarray,
    ) -> np.ndarray:
        """Computes the next prize indicators based on the current state.

        Args:
            next_position (float): The agent's next position in the grid.
            position (float): The agent's current position before moving.
            next_status (int): The next status intensity of the environment after the move.
            prize_indicators (np.ndarray): The current array of prize indicators before the move.

        Returns:
            np.ndarray: An updated array of prize indicators.
        """
        if int(next_position) == self.length - 1:
            if int(position) == self.length - 2:
                if next_status == self.max_intensity:
                    return np.ones(self.num_prize_indicators, dtype=int)
            elif all(prize_indicators == 1):
                return prize_indicators
            else:
                return self._shift_prize_indicators(prize_indicators)
        return np.zeros(self.num_prize_indicators, dtype=int)

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
        self, next_prize_indicators: np.ndarray, action: int, position: float
    ) -> float:
        """Compute the reward based on action and prize indicators.

        Args:
            next_prize_indicators (np.ndarray): Updated prize indicators.
            action (int): Action taken by the agent.
            position (float): Agent's current position.

        Returns:
            float: Calculated reward.
        """
        if all(next_prize_indicators == 1) and int(position) == self.length - 1:
            return 3.0
        return 0.0 if action == 0 else -1.0

    def _get_observation(self) -> np.ndarray:
        """Get the current observation with optional offsets.

        Returns:
            np.ndarray: The current observation.
        """
        if self.has_state_offset:
            obs = self.state.copy()
            obs[0] += self.position_offset
            obs[1] += self.status_indicator_offset
            obs[2:] += self.prize_indicator_offsets
            return obs
        return self.state.copy()

    def set_state(self, state: np.ndarray, previous_status: int) -> None:
        """Set the environment state.

        Args:
            state (np.ndarray): The new state.
            previous_status (int): Previous status intensity.
        """
        self.state = state
        self.previous_status = previous_status

    def render(self) -> None:
        """Renders the environment using Pygame and resets on 'r' press."""
        if self.render_mode != "human":
            return

        env_name = self.metadata.get("environment_name")
        top_area_height = 200 if env_name == "bbi" else 100
        bottom_area_height = 50
        width = self.margin * 2 + self.cell_size * self.length
        height = top_area_height + self.margin + self.cell_size + bottom_area_height

        # Initialize Pygame screen if not done
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(f"{env_name or 'GoRight'} Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)
            # Load images
            self.lamp_on = pygame.image.load(
                "bbi/environments/resources/lamp_on.png"
            ).convert_alpha()
            self.lamp_off = pygame.image.load(
                "bbi/environments/resources/lamp_off.png"
            ).convert_alpha()
            self.robot = pygame.image.load(
                "bbi/environments/resources/robot.png"
            ).convert_alpha()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    self.step(0)
                elif event.key == K_RIGHT:
                    self.step(1)
                # elif event.key == pygame.K_r:
                #     # Call reset and reassign state/variables if needed
                #     obs, info = self.reset()
                #     # Ensure state reflects the reset state:
                #     # self.state should now be the initial state set in reset()

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Extract state info after any possible reset
        position = int(self.state[0])
        current_status = int(self.state[1])
        prize_indicators = self.state[2:]
        prev_status = (
            int(self.previous_status) if self.previous_status is not None else 0
        )

        # Draw the grid
        grid_top = 100
        grid_left = self.margin
        for i in range(self.length):
            rect_x = grid_left + i * self.cell_size
            rect_y = grid_top
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (rect_x, rect_y, self.cell_size, self.cell_size),
                2,
            )

        # Draw the agent robot
        agent_x = grid_left + position * self.cell_size + self.cell_size // 4
        agent_y = grid_top + self.cell_size // 4
        self.screen.blit(self.robot, (agent_x, agent_y))

        # Draw last action arrow if any
        if self.last_action is not None and self.last_pos is not None:
            arr_x = (
                grid_left + int(self.last_pos) * self.cell_size + self.cell_size // 2
            )
            arr_y = grid_top + self.cell_size // 2
            if self.last_action == 0:  # Left action
                pygame.draw.polygon(
                    self.screen,
                    (255, 0, 0),
                    [
                        (arr_x - self.cell_size // 2, arr_y),
                        (arr_x - self.cell_size // 2 + 10, arr_y + 10),
                        (arr_x - self.cell_size // 2 + 10, arr_y - 10),
                    ],
                )
            else:  # Right action
                pygame.draw.polygon(
                    self.screen,
                    (255, 0, 0),
                    [
                        (arr_x + self.cell_size // 2, arr_y),
                        (arr_x + self.cell_size // 2 - 10, arr_y + 10),
                        (arr_x + self.cell_size // 2 - 10, arr_y - 10),
                    ],
                )

        def status_color(value, is_text=False):
            # Simple color mapping
            if not is_text:
                if value == 0:
                    return "#1B1D1F"
                elif value == 5:
                    return "#454C53"
                elif value == 10:
                    return "#C9CDD2"
            else:
                if value == 0:
                    return "#F7F8F9"
                elif value == 5:
                    return "#F7F8F9"
                elif value == 10:
                    return "#26282B"

        # Draw status indicators
        box_size = 40
        status_x = self.margin
        status_y = 20
        if env_name == "GoRight":
            # Prev status
            pygame.draw.rect(
                self.screen,
                status_color(prev_status),
                (status_x, status_y, box_size, box_size),
            )
            prev_text = self.font.render(
                str(prev_status), True, status_color(prev_status, True)
            )
            self.screen.blit(
                prev_text,
                (
                    status_x + (box_size - prev_text.get_width()) / 2,
                    status_y + (box_size - prev_text.get_height()) / 2,
                ),
            )
            prev_label = self.font.render("Prev", True, (0, 0, 0))
            self.screen.blit(prev_label, (status_x, status_y + box_size + 5))

            # Curr status
            pygame.draw.rect(
                self.screen,
                status_color(current_status),
                (status_x + 50, status_y, box_size, box_size),
            )
            curr_text = self.font.render(
                str(current_status), True, status_color(current_status, True)
            )
            self.screen.blit(
                curr_text,
                (
                    status_x + 50 + (box_size - curr_text.get_width()) / 2,
                    status_y + (box_size - curr_text.get_height()) / 2,
                ),
            )
            curr_label = self.font.render("Curr", True, (0, 0, 0))
            self.screen.blit(curr_label, (status_x + 50, status_y + box_size + 5))
        else:
            # Only current status
            pygame.draw.rect(
                self.screen,
                status_color(current_status),
                (status_x, status_y, box_size, box_size),
            )
            curr_text = self.font.render(
                str(current_status), True, status_color(current_status, True)
            )
            self.screen.blit(
                curr_text,
                (
                    status_x + (box_size - curr_text.get_width()) / 2,
                    status_y + (box_size - curr_text.get_height()) / 2,
                ),
            )
            curr_label = self.font.render("Curr", True, (0, 0, 0))
            self.screen.blit(curr_label, (status_x, status_y + box_size + 5))

        # Draw prize indicators
        lamps_x = width - self.margin - self.num_prize_indicators * 40
        lamp_y = 30
        for i, val in enumerate(prize_indicators):
            lamp_x = lamps_x + i * 40
            img = self.lamp_on if val > 0.5 else self.lamp_off
            self.screen.blit(img, (lamp_x, lamp_y))

        # Show rewards and actions
        reward_label = self.font.render(
            f"Last Reward: {self.last_reward}", True, (0, 0, 0)
        )
        self.screen.blit(reward_label, (self.margin, 170))
        cum_reward_label = self.font.render(
            f"Cumulative Reward: {self.total_reward}", True, (0, 0, 0)
        )
        self.screen.blit(cum_reward_label, (self.margin, 190))
        actions_label = self.font.render(
            f"Actions taken: {self.action_count}", True, (0, 0, 0)
        )
        self.screen.blit(actions_label, (self.margin, 210))

        # If bbi, show bounding boxes
        if env_name == "bbi":
            bbox_y_start = 250
            if (
                hasattr(self, "state_bounding_box")
                and self.state_bounding_box is not None
            ):
                # pygame.draw.rect(self.screen, (220,220,220), (self.margin, bbox_y_start, 270, 50), 2)
                state_min = self.state_bounding_box[0]
                state_max = self.state_bounding_box[1]
                state_min_str = "Lower State: " + np.array2string(
                    state_min, precision=2, separator=",", suppress_small=True
                )
                state_max_str = "Upper State: " + np.array2string(
                    state_max, precision=2, separator=",", suppress_small=True
                )
                min_text = self.font.render(state_min_str, True, (0, 0, 0))
                max_text = self.font.render(state_max_str, True, (0, 0, 0))
                self.screen.blit(min_text, (self.margin + 5, bbox_y_start + 5))
                self.screen.blit(max_text, (self.margin + 5, bbox_y_start + 25))

            if (
                hasattr(self, "reward_bounding_box")
                and self.reward_bounding_box is not None
            ):
                # pygame.draw.rect(self.screen, (220,220,220), (self.margin+280, bbox_y_start, 340, 50), 2)
                reward_min, reward_max = self.reward_bounding_box
                reward_str = f"Reward Boundaries: [{reward_min}, {reward_max}]"
                reward_text = self.font.render(reward_str, True, (0, 0, 0))
                self.screen.blit(reward_text, (self.margin + 295, bbox_y_start + 15))

        pygame.display.flip()
        self.clock.tick(10)


def close(self):
    """Closes the rendering window and quits Pygame."""
    if self.screen is not None:
        pygame.quit()
        self.screen = None
        self.clock = None
