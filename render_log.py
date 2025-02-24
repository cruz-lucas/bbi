"""
Render an interactive animation of Q–value evolution from a training log file using Matplotlib.

The log file must contain lines like:
  2025-02-11 ... DEBUG [bbi.agent] Update for state (0, 2, 0, 0), action 1: ... updated Q=-0.04000000000000001

This script extracts, for each update, the global training step, the state (a tuple of 4 integers),
the action (0 or 1), and the updated Q value. The state is interpreted as:
    s = (position, status, prize1, prize2)
with
    position ∈ {0, …, 10},
    status ∈ {0, 1, 2},
    prize1 ∈ {0, 1},
    prize2 ∈ {0, 1}.

For each combination of status and prize indicators (12 groups total), the visualization shows two rows
(one for action 0 and one for action 1) with 11 columns (positions). The cell colors are determined by a blue–to–red
diverging colormap (with 0 as white). Interactive sliders let you select the training step and adjust vmin/vmax while
keeping 0 centered. A play/pause button is also provided.
"""

import re
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import Button, Slider


def parse_log_file(filename: str):
    """
    Parse the log file to extract Q–value updates.

    Each update line should match:
      "Update for state (<state>), action <action>: ... updated Q=<q_value>"
    Global step numbers are taken from lines containing "global step <number>".

    Args:
        filename (str): Path to the log file.

    Returns:
        List of tuples (global_step, state, action, q_value), where state is a tuple of ints.
    """
    update_pattern = re.compile(
        r"Update for state \((.*?)\), action (\d+):.*updated Q=([-\d.eE]+)"
    )
    gs_pattern = re.compile(r"global step (\d+)")
    timeline = []
    current_global_step = 0

    with open(filename) as f:
        for line in f:
            gs_match = gs_pattern.search(line)
            if gs_match:
                current_global_step = int(gs_match.group(1))
            m = update_pattern.search(line)
            if m:
                state_str = m.group(1)  # e.g. "0, 2, 0, 0"
                action = int(m.group(2))
                q_val = float(m.group(3))
                # Parse state tuple (position, status, prize1, prize2)
                state = tuple(int(x.strip()) for x in state_str.split(","))
                timeline.append((current_global_step, state, action, q_val))
    return timeline


def build_q_evolution(timeline, num_positions: int = 11, initial_value: float = 0.0):
    """
    Organize the Q–value updates into a timeline for each state–action cell.

    The state is defined as (position, status, prize1, prize2). There are:
      - 11 positions,
      - status: 0, 1, 2,
      - prize1: 0, 1,
      - prize2: 0, 1,
    and for each such state, two actions (0 and 1).

    Returns:
        dict: Mapping keys (pos, status, prize1, prize2, action) to a sorted list of (global_step, q_value).
    """
    q_data = {}
    statuses = [0, 1, 2]
    prizes = [0, 1]
    for pos in range(num_positions):
        for status in statuses:
            for p1 in prizes:
                for p2 in prizes:
                    for action in [0, 1]:
                        q_data[(pos, status, p1, p2, action)] = [(0, initial_value)]
    for gs, state, action, q_val in timeline:
        key = (state[0], state[1], state[2], state[3], action)
        q_data[key].append((gs, q_val))
    # Sort each list by global_step.
    for key in q_data:
        q_data[key].sort(key=lambda x: x[0])
    return q_data


def get_q_at_step(q_list, step):
    """
    Get the latest Q value from a sorted list of (global_step, q_value) for a given training step.

    Args:
        q_list: List of (global_step, q_value).
        step: Global training step.

    Returns:
        The Q value at the given step (or the most recent update before that step).
    """
    value = q_list[0][1]
    for gs, q in q_list:
        if gs <= step:
            value = q
        else:
            break
    return value


def main():
    """Main routine to parse the log file, build the Q evolution timeline, and render the interactive animation."""
    if len(sys.argv) < 2:
        print("Usage: python render_q_evolution_matplotlib.py <log_file>")
        sys.exit(1)
    log_file = sys.argv[1]
    timeline = parse_log_file(log_file)
    if not timeline:
        print("No Q update entries found in the log file.")
        sys.exit(1)
    q_data = build_q_evolution(timeline, num_positions=11, initial_value=0.0)
    total_steps = max(gs for gs, s, a, q in timeline)

    # Define grid parameters.
    num_positions = 11
    statuses = [0, 1, 2]
    prize_combinations = [(p1, p2) for p1 in [0, 1] for p2 in [0, 1]]
    # There are 12 groups (3 statuses x 4 prize combinations)
    grid_cols = 4
    grid_rows = 3

    # Create a figure with 3 rows x 4 columns of subplots.
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(16, 12))
    # Reserve space at the bottom for three sliders (training step, vmin, vmax) and a play/pause button.
    plt.subplots_adjust(bottom=0.25, top=0.92, wspace=0.4, hspace=0.4)

    # Use a diverging norm that forces 0 to be white.
    global_vmin, global_vmax = -5, 30
    norm = TwoSlopeNorm(vmin=global_vmin, vcenter=0, vmax=global_vmax)

    # For each group (combination of status and prize indicators), prepare a 2x11 image.
    group_keys = []
    for status in statuses:
        for combo in prize_combinations:
            group_keys.append((status, combo[0], combo[1]))

    group_images = {}
    group_data = {}
    cell_rows = 2  # Two actions.
    cell_cols = num_positions

    # Create subplots and initial images.
    for i, key in enumerate(group_keys):
        row = i // grid_cols
        col = i % grid_cols
        ax = axs[row, col]
        data = np.zeros((cell_rows, cell_cols))
        # Fill with initial data (training step 0).
        for pos in range(cell_cols):
            for action in range(cell_rows):
                state_key = (pos, key[0], key[1], key[2], action)
                data[action, pos] = get_q_at_step(q_data[state_key], 0)
        group_data[key] = data
        im = ax.imshow(data, cmap="RdBu", norm=norm, aspect="auto")
        ax.set_title(f"Status {key[0]} | Prize {key[1]}{key[2]}")
        ax.set_xticks(range(cell_cols))
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Action 0", "Action 1"])
        group_images[key] = im

    # Create a global colorbar using one of the images.
    cbar = fig.colorbar(list(group_images.values())[0], ax=axs.ravel().tolist(), orientation="vertical")
    cbar.set_label("Q value (0 = white)")

    # Add interactive slider for training step.
    ax_step = plt.axes([0.15, 0.18, 0.65, 0.03])
    step_slider = Slider(ax_step, "Step", 0, total_steps, valinit=0, valfmt="%d")

    # Add interactive sliders for vmin and vmax.
    ax_vmin = plt.axes([0.15, 0.12, 0.3, 0.03])
    vmin_slider = Slider(ax_vmin, "vmin", -100, 0, valinit=global_vmin, valfmt="%.1f")
    ax_vmax = plt.axes([0.55, 0.12, 0.3, 0.03])
    vmax_slider = Slider(ax_vmax, "vmax", 0, 100, valinit=global_vmax, valfmt="%.1f")

    # Add play/pause button.
    ax_button = plt.axes([0.82, 0.04, 0.1, 0.04])
    play_button = Button(ax_button, "Play/Pause")
    playing = [False]  # Mutable flag.

    def update_plots(val):
        """Update each subplot for the current training step and update the normalization."""
        step = int(step_slider.val)
        # Update normalization with current slider values.
        current_vmin = vmin_slider.val
        current_vmax = vmax_slider.val
        new_norm = TwoSlopeNorm(vmin=current_vmin, vcenter=0, vmax=current_vmax)
        for key in group_keys:
            data = np.zeros((cell_rows, cell_cols))
            for pos in range(cell_cols):
                for action in range(cell_rows):
                    state_key = (pos, key[0], key[1], key[2], action)
                    data[action, pos] = get_q_at_step(q_data[state_key], step)
            group_data[key] = data
            group_images[key].set_data(data)
            group_images[key].set_norm(new_norm)
        cbar.mappable.set_norm(new_norm)
        fig.canvas.draw_idle()

    step_slider.on_changed(update_plots)
    vmin_slider.on_changed(update_plots)
    vmax_slider.on_changed(update_plots)

    def toggle_play(event):
        """Toggle play/pause state."""
        playing[0] = not playing[0]

    play_button.on_clicked(toggle_play)

    def animate():
        """Timer callback: if playing, advance the slider by one step."""
        if playing[0]:
            current = int(step_slider.val)
            new_val = current + 500
            if new_val > total_steps:
                new_val = 0
            step_slider.set_val(new_val)

    timer = fig.canvas.new_timer(interval=100)  # 100 milliseconds per frame.
    timer.add_callback(animate)
    timer.start()

    plt.show()


if __name__ == "__main__":
    main()
