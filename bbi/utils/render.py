import pygame
from pygame.locals import K_ESCAPE, K_LEFT, K_RIGHT, KEYDOWN, MOUSEBUTTONDOWN, K_m, K_r

from bbi.environments import GoRight
from bbi.models import BBI, ExpectationModel, SamplingModel
from bbi.utils.constants import (
    BOTTOM_AREA_HEIGHT,
    CELL_SIZE,
    GRID_TOP,
    MARGIN,
    TOP_AREA_HEIGHT,
)

MODEL_CLASSES = [GoRight, SamplingModel, ExpectationModel, BBI]


_screen = None
_clock = None
_font = None
_strong_font = None
_robot_img = None
_lamp_on_img = None
_lamp_off_img = None
_initialized = False  # Tracks if we've already done one-time pygame initialization.

# --- Text input boxes and "Set State" button ---
# We’ll store metadata for each text box (position, size, current text, etc.).
TEXT_BOXES = [
    {
        "name": "pos",
        "rect": pygame.Rect(MARGIN, 0, 100, 30),
        "text": "",
        "active": False,
        "placeholder": "Position",
    },
    {
        "name": "prev",
        "rect": pygame.Rect(MARGIN + 110, 0, 100, 30),
        "text": "",
        "active": False,
        "placeholder": "PrevSt",
    },
    {
        "name": "curr",
        "rect": pygame.Rect(MARGIN + 220, 0, 100, 30),
        "text": "",
        "active": False,
        "placeholder": "CurrSt",
    },
    {
        "name": "prizes",
        "rect": pygame.Rect(MARGIN + 330, 0, 220, 30),
        "text": "",
        "active": False,
        "placeholder": "Prizes (comma)",
    },
]

# Button to apply the state
SET_STATE_BUTTON = {
    "rect": pygame.Rect(MARGIN + 580, 0, 100, 30),
    "text": "Set State",
}


def _init_pygame(width: int, height: int, env_name: str = "Environment", force: bool = False) -> None:
    """
    One-time initialization for Pygame. Sets up display, clock, font, images, etc.
    Stores references in module-level variables.
    """
    global _screen, _clock, _font, _strong_font, _robot_img, _lamp_on_img, _lamp_off_img, _initialized

    # Prevent re-initializing if already done
    if _initialized and not force:
        return

    pygame.init()
    pygame.display.set_caption(env_name)

    _screen = pygame.display.set_mode((width, height))
    _clock = pygame.time.Clock()
    _font = pygame.font.SysFont("Arial", 20)
    _strong_font = pygame.font.SysFont("Arial", 24, bold=True)

    # Load images just once
    _robot_img = pygame.image.load(
        "bbi/environments/resources/robot.png"
    ).convert_alpha()
    _lamp_on_img = pygame.image.load(
        "bbi/environments/resources/lamp_on.png"
    ).convert_alpha()
    _lamp_off_img = pygame.image.load(
        "bbi/environments/resources/lamp_off.png"
    ).convert_alpha()

    _initialized = True


def _status_color(value: int, is_text: bool = False) -> str:
    """
    Returns a hex color string based on the status indicator value.
    If is_text == True, returns a text-friendly color.
    """
    if not is_text:
        return {0: "#1B1D1F", 5: "#454C53", 10: "#C9CDD2"}.get(value, "#C9CDD2")
    return {0: "#F7F8F9", 5: "#F7F8F9", 10: "#26282B"}.get(value, "#F7F8F9")


def _draw_action_arrow(env: GoRight) -> None:
    """
    Draws a small red arrow (triangle) showing the last action the agent took.
    """
    if not env.tracker.history:
        return

    last_entry = env.tracker.history[-1]
    last_pos = last_entry.state.position
    last_action = last_entry.action  # 0 or 1

    # Center of the cell
    arr_x = MARGIN + int(last_pos) * CELL_SIZE + CELL_SIZE // 2
    arr_y = GRID_TOP + CELL_SIZE // 2
    color = (255, 0, 0)

    if last_action == 0:
        # Left arrow
        pygame.draw.polygon(
            _screen,
            color,
            [
                (arr_x - CELL_SIZE // 2, arr_y),
                (arr_x - CELL_SIZE // 2 + 10, arr_y + 10),
                (arr_x - CELL_SIZE // 2 + 10, arr_y - 10),
            ],
        )
    else:
        # Right arrow
        pygame.draw.polygon(
            _screen,
            color,
            [
                (arr_x + CELL_SIZE // 2, arr_y),
                (arr_x + CELL_SIZE // 2 - 10, arr_y + 10),
                (arr_x + CELL_SIZE // 2 - 10, arr_y - 10),
            ],
        )


def _draw_status_indicators(env: GoRight, env_name: str) -> None:
    """
    Draws rectangles (boxes) showing the previous and current status indicators,
    and labels them "Prev" and "Curr".
    """
    prev_status = env.state.previous_status_indicator
    current_status = env.state.current_status_indicator

    box_size = 40
    status_x = MARGIN
    status_y = 20

    if env_name == "GoRight":
        # Draw 'Prev'
        pygame.draw.rect(
            _screen,
            _status_color(prev_status),
            (status_x, status_y, box_size, box_size),
        )
        prev_text = _font.render(
            str(prev_status), True, _status_color(prev_status, True)
        )
        _screen.blit(
            prev_text,
            (
                status_x + (box_size - prev_text.get_width()) / 2,
                status_y + (box_size - prev_text.get_height()) / 2,
            ),
        )
        prev_label = _font.render("Prev", True, (0, 0, 0))
        _screen.blit(prev_label, (status_x, status_y + box_size + 5))

        # Draw 'Curr'
        pygame.draw.rect(
            _screen,
            _status_color(current_status),
            (status_x + 50, status_y, box_size, box_size),
        )
        curr_text = _font.render(
            str(current_status), True, _status_color(current_status, True)
        )
        _screen.blit(
            curr_text,
            (
                status_x + 50 + (box_size - curr_text.get_width()) / 2,
                status_y + (box_size - curr_text.get_height()) / 2,
            ),
        )
        curr_label = _font.render("Curr", True, (0, 0, 0))
        _screen.blit(curr_label, (status_x + 50, status_y + box_size + 5))

    else:
        # Only 'Curr'
        pygame.draw.rect(
            _screen,
            _status_color(current_status),
            (status_x, status_y, box_size, box_size),
        )
        curr_text = _font.render(
            str(current_status), True, _status_color(current_status, True)
        )
        _screen.blit(
            curr_text,
            (
                status_x + (box_size - curr_text.get_width()) / 2,
                status_y + (box_size - curr_text.get_height()) / 2,
            ),
        )
        curr_label = _font.render("Curr", True, (0, 0, 0))
        _screen.blit(curr_label, (status_x, status_y + box_size + 5))


def _draw_prize_indicators(env: GoRight, screen_width: int) -> None:
    """
    Draws small lamp images on the right side of the screen indicating
    the 'prize' lights (if turned on).
    """
    if _lamp_on_img is None or _lamp_off_img is None:
        return  # No images loaded or non-human render mode

    lamp_y = 30
    lamps_x = screen_width - MARGIN - env.num_prize_indicators * 40

    for i, val in enumerate(env.state.prize_indicators):
        lamp_x = lamps_x + i * 40
        img = _lamp_on_img if val > 0.5 else _lamp_off_img
        _screen.blit(img, (lamp_x, lamp_y))


def _draw_info_text(env: GoRight) -> None:
    """
    Renders textual info about the last reward, total reward, and how many
    actions have been taken so far.
    """
    if not env.tracker.history:
        last_reward = 0.0
        total_reward = 0.0
        action_count = 0
    else:
        last_entry = env.tracker.history[-1]
        last_reward = last_entry.reward
        total_reward = env.tracker.total_reward
        action_count = env.tracker.action_count

    reward_label = _font.render(f"Last Reward: {last_reward}", True, (0, 0, 0))
    _screen.blit(reward_label, (MARGIN, 170))

    cum_reward_label = _font.render(
        f"Cumulative Reward: {total_reward}", True, (0, 0, 0)
    )
    _screen.blit(cum_reward_label, (MARGIN, 190))

    actions_label = _font.render(f"Actions taken: {action_count}", True, (0, 0, 0))
    _screen.blit(actions_label, (MARGIN, 210))


def _draw_state_input_menu(env: GoRight, width: int, height: int):
    """
    Draws the text boxes and the "Set State" button at the bottom of the screen.
    """
    base_y = height - BOTTOM_AREA_HEIGHT + 10  # Ensures alignment at the bottom

    # Draw each text box with a rectangle, placeholder or typed text
    for tb in TEXT_BOXES:
        tb["rect"].y = base_y
        color = (
            pygame.Color("lightskyblue3") if tb["active"] else pygame.Color("gray80")
        )
        pygame.draw.rect(_screen, color, tb["rect"], border_radius=5)

        # Render text or placeholder
        display_text = tb["text"] if tb["text"] else f"<{tb['placeholder']}>"
        txt_surface = _font.render(display_text, True, (0, 0, 0))
        _screen.blit(
            txt_surface,
            (
                tb["rect"].x + 5,
                tb["rect"].y + (tb["rect"].height - txt_surface.get_height()) // 2,
            ),
        )

    # Draw the "Set State" button
    SET_STATE_BUTTON["rect"].y = base_y
    pygame.draw.rect(
        _screen, pygame.Color("lightgreen"), SET_STATE_BUTTON["rect"], border_radius=5
    )
    btn_text_surf = _font.render(SET_STATE_BUTTON["text"], True, (0, 0, 0))
    _screen.blit(
        btn_text_surf,
        (
            SET_STATE_BUTTON["rect"].x
            + (SET_STATE_BUTTON["rect"].width - btn_text_surf.get_width()) // 2,
            SET_STATE_BUTTON["rect"].y
            + (SET_STATE_BUTTON["rect"].height - btn_text_surf.get_height()) // 2,
        ),
    )



def _handle_text_input_event(event):
    """
    Handles KEYDOWN events for the active text box. Allows letters, digits,
    punctuation, backspace, etc.
    """
    for tb in TEXT_BOXES:
        if tb["active"]:
            if event.key == pygame.K_BACKSPACE:
                tb["text"] = tb["text"][:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                # (Optional) Pressing Enter could finalize input or move to next field
                pass
            else:
                # Append character if it's a printable one
                # Filter out special keys that we don't want as text
                if 32 <= event.key < 127:  # Basic printable ASCII range
                    tb["text"] += event.unicode
            break


def _handle_mouse_click_event(pos, env: GoRight):
    """
    Handles MOUSEBUTTONDOWN events. We detect if the user clicked:
    1. Inside a text box => activate that box
    2. Inside the "Set State" button => parse & apply new state
    """
    clicked_box = False
    for tb in TEXT_BOXES:
        if tb["rect"].collidepoint(pos):
            # Activate this box, deactivate others
            for other in TEXT_BOXES:
                other["active"] = False
            tb["active"] = True
            clicked_box = True
        else:
            tb["active"] = False

    # Check if "Set State" button was clicked
    if SET_STATE_BUTTON["rect"].collidepoint(pos):
        _apply_new_state(env)
        return

    # If we didn’t click any text box or button, deactivate all
    if not clicked_box:
        for tb in TEXT_BOXES:
            tb["active"] = False


def _apply_new_state(env: GoRight):
    """
    Attempts to parse the user input from the text boxes and set
    the environment's state accordingly. Resets env and overwrites state.
    """
    # Extract the text from each box
    pos_text = TEXT_BOXES[0]["text"]
    prev_text = TEXT_BOXES[1]["text"]
    curr_text = TEXT_BOXES[2]["text"]
    prizes_text = TEXT_BOXES[3]["text"]

    try:
        new_pos = int(pos_text)
        new_prev = int(prev_text)
        new_curr = int(curr_text)
        # For prizes, we expect something like "0,1" => [0, 1]
        # or "1,1" => [1,1]. Let’s parse:
        if prizes_text.strip():
            # Split by commas
            str_vals = [x.strip() for x in prizes_text.split(",")]
            parsed_prizes = list(map(int, str_vals))
        else:
            # If nothing typed in, default to zeros
            parsed_prizes = [0] * env.num_prize_indicators

        # In case user typed more or fewer than needed
        # we can slice or pad the list
        if len(parsed_prizes) < env.num_prize_indicators:
            # Pad with 0s
            parsed_prizes += [0] * (env.num_prize_indicators - len(parsed_prizes))
        elif len(parsed_prizes) > env.num_prize_indicators:
            parsed_prizes = parsed_prizes[: env.num_prize_indicators]

        # Now reset the env and set the new state
        env.reset()  # ensures tracker etc. is cleared
        env.state.set_state(
            position=new_pos,
            previous_status_indicator=new_prev,
            current_status_indicator=new_curr,
            prize_indicators=parsed_prizes,
        )

        print(
            f"State updated: pos={new_pos}, prev={new_prev}, curr={new_curr}, prizes={parsed_prizes}"
        )
    except ValueError:
        print(
            "Could not parse state from text input. Make sure to enter valid integers."
        )


def _draw_bounding_boxes(env: BBI) -> None:
    """
    Draws bounding box information with extra space for the BBI environment.
    """
    bbox_y_start = 250
    _margin = MARGIN

    if env.bounding_box is not None:
        state_min = env.bounding_box.state_lower_bound.get_state()
        state_max = env.bounding_box.state_upper_bound.get_state()

        mask = [True] * len(state_max)
        mask[1] = False

        previous_state = env.tracker.history[-1].state.get_state()

        # Format numbers to 2 decimal places and align them
        state_min_fmt = [f"{i:>6.2f}" for i in state_min[mask]]
        state_max_fmt = [f"{i:>6.2f}" for i in state_max[mask]]
        state_fmt = [f"{i:>6.2f}" for i in previous_state[mask]]

        # Create strings with aligned numbers
        state_min_str = f"State Lower Bound:    {' '.join(state_min_fmt)}"
        state_max_str = f"State Upper Bound:   {' '.join(state_max_fmt)}"
        state_str = f"t+1 State:                  {' '.join(state_fmt)}"

        # Render the strings as text
        min_text = _font.render(state_min_str, True, (0, 0, 0))
        state_text = _font.render(state_str, True, (0, 0, 0))
        max_text = _font.render(state_max_str, True, (0, 0, 0))
        title = _strong_font.render("t+2 Bounding Box", True, (0, 0, 0))

        interval = 30
        # Display the text on the screen
        _screen.blit(title, (_margin, bbox_y_start))
        _screen.blit(max_text, (_margin, bbox_y_start + interval))
        _screen.blit(state_text, (_margin, bbox_y_start + 2*interval))
        _screen.blit(min_text, (_margin, bbox_y_start + 3*interval))

        # Render and display the reward bounds
        reward_min = env.bounding_box.reward_lower_bound
        reward_max = env.bounding_box.reward_upper_bound
        reward_str = f"Reward Bounds:     {reward_min:.2f}, {reward_max:.2f}"
        reward_text = _font.render(reward_str, True, (0, 0, 0))
        _screen.blit(reward_text, (_margin, bbox_y_start + 5*interval))  # Extra spacing





        rolling_bb_min = env.rolling_bounding_box.state_lower_bound.get_state()
        rolling_bb_max = env.rolling_bounding_box.state_upper_bound.get_state()

        # Format numbers to 2 decimal places and align them
        rolling_bb_min_fmt = [f"{i:>6.2f}" for i in rolling_bb_min[mask]]
        rolling_bb_max_fmt = [f"{i:>6.2f}" for i in rolling_bb_max[mask]]

        # Create strings with aligned numbers
        rolling_bb_min_str = f"State Lower Bound:    {' '.join(rolling_bb_min_fmt)}"
        rolling_bb_max_str = f"State Upper Bound:   {' '.join(rolling_bb_max_fmt)}"
        # state_str = f"t+1     State:            {' '.join(state_fmt)}"
        title = _strong_font.render("Rolling (t+h) Bounding Box", True, (0, 0, 0))

        # Render the strings as text
        min_text = _font.render(rolling_bb_min_str, True, (0, 0, 0))
        max_text = _font.render(rolling_bb_max_str, True, (0, 0, 0))

        # Display the text on the screen
        _screen.blit(title, (_margin, bbox_y_start + 7*interval))
        _screen.blit(max_text, (_margin, bbox_y_start + 8*interval))
        _screen.blit(min_text, (_margin, bbox_y_start + 9*interval))


        previous_bb = env.bbi_tracker.history[-2]
        prev_bb_min = previous_bb.state_lower_bound.get_state()
        prev_bb_max = previous_bb.state_upper_bound.get_state()

        # Format numbers to 2 decimal places and align them
        prev_bb_min_fmt = [f"{i:>6.2f}" for i in prev_bb_min[mask]]
        prev_bb_max_fmt = [f"{i:>6.2f}" for i in prev_bb_max[mask]]

        # Create strings with aligned numbers
        prev_bb_min_str = f"State Lower Bound:    {' '.join(prev_bb_min_fmt)}"
        prev_bb_max_str = f"State Upper Bound:   {' '.join(prev_bb_max_fmt)}"

        prev_bb_min_text = _font.render(prev_bb_min_str, True, (0, 0, 0))
        prev_bb_max_text = _font.render(prev_bb_max_str, True, (0, 0, 0))

        title = _strong_font.render("t+h-1 Bounding Box", True, (0, 0, 0))

        _screen.blit(title, (_margin, bbox_y_start + 11*interval))
        _screen.blit(prev_bb_max_text, (_margin, bbox_y_start + 12*interval))
        _screen.blit(prev_bb_min_text, (_margin, bbox_y_start + 13*interval))


        # Render and display the reward bounds
        reward_min = env.rolling_bounding_box.reward_lower_bound
        reward_max = env.rolling_bounding_box.reward_upper_bound
        reward_str = f"Reward Bounds:     {reward_min:.2f}, {reward_max:.2f}"
        reward_text = _font.render(reward_str, True, (0, 0, 0))
        _screen.blit(reward_text, (_margin, bbox_y_start + 15*interval))  # Extra spacing




def render_env(env: GoRight, model_classes, current_model_index: int):
    """
    Main rendering function. Initializes Pygame if needed, draws the environment
    grid, agent, and status info, and handles keyboard/mouse events for stepping,
    model switching, resetting, and setting new state.
    """
    env_name = env.metadata.get("environment_name", "Environment")

    # Calculate required window size
    base_height = TOP_AREA_HEIGHT + MARGIN + CELL_SIZE + BOTTOM_AREA_HEIGHT + 20
    extra_height = 485 if env_name == "BBI" else 0  # Add extra height for bounding boxes
    width = MARGIN * 2 + CELL_SIZE * env.length
    height = base_height + extra_height

    # Initialize Pygame and global references if not done
    global _screen
    if _screen is None or _screen.get_size() != (width, height):
        _init_pygame(width, height, env_name, force=True)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        elif event.type == MOUSEBUTTONDOWN:
            _handle_mouse_click_event(event.pos, env)
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                raise SystemExit
            elif event.key == K_LEFT:
                env.step(0)  # Step left
            elif event.key == K_RIGHT:
                env.step(1)  # Step right

            elif event.key == K_m:
                # Cycle to the next model
                current_model_index = (current_model_index + 1) % len(model_classes)
                env = model_classes[current_model_index]()

                env_name = env.metadata.get("environment_name", "Environment")
                pygame.display.set_caption(env_name)

                env.reset()
            elif event.key == K_r:
                # Reset the current environment
                env.reset()
            else:
                # Handle text input in the active box
                _handle_text_input_event(event)

    # Clear screen with white color
    _screen.fill((255, 255, 255))

    # Draw the grid cells
    grid_left = MARGIN
    for i in range(env.length):
        rect_x = grid_left + i * CELL_SIZE
        rect_y = GRID_TOP
        pygame.draw.rect(_screen, (0, 0, 0), (rect_x, rect_y, CELL_SIZE, CELL_SIZE), 2)

    # Draw the agent (robot image) if loaded
    if _robot_img is not None and env.state:
        agent_x = grid_left + int(env.state.position) * CELL_SIZE + CELL_SIZE // 4
        agent_y = GRID_TOP + CELL_SIZE // 4
        _screen.blit(_robot_img, (agent_x, agent_y))

    # Draw the arrow showing the last action
    _draw_action_arrow(env)

    # Draw status indicator boxes
    _draw_status_indicators(env, env_name)

    # Draw prize indicators (lamps) on the right
    _draw_prize_indicators(env, width)

    # Draw textual info (reward, total, actions)
    _draw_info_text(env)

    # Draw text input boxes + "Set State" button
    _draw_state_input_menu(env, width, height)

    if env_name == "BBI":
        _draw_bounding_boxes(env)

    # Flip the display to show changes
    pygame.display.flip()

    # Limit FPS
    _clock.tick(10)

    return env, current_model_index


if __name__ == "__main__":
    # Example usage: keep cycling in a loop, rendering frames.
    current_model_index = 0
    env = MODEL_CLASSES[current_model_index]()
    env.reset()

    running = True
    while running:
        env, current_model_index = render_env(env, MODEL_CLASSES, current_model_index)
