from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from pygame.locals import K_LEFT, K_RIGHT, KEYDOWN


class BaseEnv(gym.Env):
    def __init__(self) -> None:
        """"""
        super().__init__()

        # Rendering related attributes
        self.screen = None
        self.clock = None
        self.font = None

        self.cell_size = 60
        self.margin = 50
        self.bottom_area_height = 50
        self.top_area_height = 100
        self.grid_top = 100

        # Image sprites for lamps
        self.lamp_on = None
        self.lamp_off = None
        self.robot = None

    def seed(self, seed: Optional[int] = None) -> None:
        """Sets the seed for the environment's random number generator.

        Args:
            seed: The seed value.
        """
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Abstract method: Implement in subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_observation(self) -> np.ndarray:
        """Abstract method: Implement in subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_state(self, state: np.ndarray, previous_status: int) -> None:
        """Abstract method: Implement in subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def render(self) -> None:
        """Render the environment using Pygame."""
        if self.render_mode != "human":
            return

        env_name = self.metadata.get("environment_name")
        width = self.margin * 2 + self.cell_size * self.length
        height = (
            self.top_area_height
            + self.margin
            + self.cell_size
            + self.bottom_area_height
        )

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(f"{env_name or 'Environment'}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)
            self.lamp_on = pygame.image.load(
                "bbi/environments/resources/lamp_on.png"
            ).convert_alpha()
            self.lamp_off = pygame.image.load(
                "bbi/environments/resources/lamp_off.png"
            ).convert_alpha()
            self.robot = pygame.image.load(
                "bbi/environments/resources/robot.png"
            ).convert_alpha()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    self.step(0)
                elif event.key == K_RIGHT:
                    self.step(1)

        self.screen.fill((255, 255, 255))

        position = int(self.state[0])
        current_status = int(self.state[1])
        prize_indicators = self.state[2:]
        prev_status = (
            int(self.previous_status) if self.previous_status is not None else 0
        )

        # Draw grid
        grid_left = self.margin
        for i in range(self.length):
            rect_x = grid_left + i * self.cell_size
            rect_y = self.grid_top
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (rect_x, rect_y, self.cell_size, self.cell_size),
                2,
            )

        # Draw agent
        agent_x = grid_left + position * self.cell_size + self.cell_size // 4
        agent_y = self.grid_top + self.cell_size // 4
        self.screen.blit(self.robot, (agent_x, agent_y))

        # Draw last action arrow
        if self.last_action is not None and self.last_pos is not None:
            self._draw_action_arrow(grid_left)

        # Draw status indicators
        self._draw_status_indicators(env_name, prev_status, current_status)

        # Draw prize indicators
        self._draw_prize_indicators(width, prize_indicators)

        # Draw info text (rewards, actions, etc.)
        self._draw_info_text()

        # Optionally draw bounding boxes for BBI
        if env_name == "BBI":
            self._draw_bounding_boxes()

        pygame.display.flip()
        self.clock.tick(10)

    def _status_color(self, value: int, is_text: bool = False) -> str:
        if not is_text:
            return {0: "#1B1D1F", 5: "#454C53", 10: "#C9CDD2"}.get(value, "#C9CDD2")
        return {0: "#F7F8F9", 5: "#F7F8F9", 10: "#26282B"}.get(value, "#F7F8F9")

    def _draw_status_indicators(
        self, env_name: str, prev_status: int, current_status: int
    ) -> None:
        box_size = 40
        status_x = self.margin
        status_y = 20

        if env_name == "GoRight":
            # Prev
            pygame.draw.rect(
                self.screen,
                self._status_color(prev_status),
                (status_x, status_y, box_size, box_size),
            )
            prev_text = self.font.render(
                str(prev_status), True, self._status_color(prev_status, True)
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

            # Curr
            pygame.draw.rect(
                self.screen,
                self._status_color(current_status),
                (status_x + 50, status_y, box_size, box_size),
            )
            curr_text = self.font.render(
                str(current_status), True, self._status_color(current_status, True)
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
            # Only Curr
            pygame.draw.rect(
                self.screen,
                self._status_color(current_status),
                (status_x, status_y, box_size, box_size),
            )
            curr_text = self.font.render(
                str(current_status), True, self._status_color(current_status, True)
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

    def _draw_prize_indicators(self, width: int, prize_indicators: np.ndarray) -> None:
        lamps_x = width - self.margin - self.num_prize_indicators * 40
        lamp_y = 30
        for i, val in enumerate(prize_indicators):
            lamp_x = lamps_x + i * 40
            img = self.lamp_on if val > 0.5 else self.lamp_off
            self.screen.blit(img, (lamp_x, lamp_y))

    def _draw_info_text(self) -> None:
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

    def _draw_action_arrow(self, grid_left: int) -> None:
        arr_x = grid_left + int(self.last_pos) * self.cell_size + self.cell_size // 2
        arr_y = self.grid_top + self.cell_size // 2
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

    def _draw_bounding_boxes(self) -> None:
        bbox_y_start = 250
        if hasattr(self, "state_bounding_box") and self.state_bounding_box is not None:
            state_min = self.state_bounding_box[0]
            state_max = self.state_bounding_box[1]
            state_min_str = "State Lower Bound: " + np.array2string(
                state_min, precision=2, separator=",", suppress_small=True
            )
            state_max_str = "State Upper Bound: " + np.array2string(
                state_max, precision=2, separator=",", suppress_small=True
            )
            min_text = self.font.render(state_min_str, True, (0, 0, 0))
            max_text = self.font.render(state_max_str, True, (0, 0, 0))
            self.screen.blit(min_text, (self.margin, bbox_y_start + 5))
            self.screen.blit(max_text, (self.margin, bbox_y_start + 25))

        if (
            hasattr(self, "reward_bounding_box")
            and self.reward_bounding_box is not None
        ):
            reward_min, reward_max = self.reward_bounding_box
            reward_str = f"Reward Bounds: [{reward_min}, {reward_max}]"
            reward_text = self.font.render(reward_str, True, (0, 0, 0))
            self.screen.blit(reward_text, (self.margin, bbox_y_start + 45))

    def close(self):
        """Closes the rendering window and quits Pygame."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
