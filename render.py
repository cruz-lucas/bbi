# from bbi.environments import GoRight
# from bbi.models import SamplingModel
# from bbi.models import ExpectationModel
# from bbi.models import BBI

# # env = GoRight(num_prize_indicators=10)
# # env = ExpectationModel()
# env = SamplingModel()
# # env = BBI()
# obs, info = env.reset()

# while True:
#     # env.set_state([0.0, 0.0, 0.0, 0.0], previous_status=0.0)
#     env.render()

import pygame
from pygame.locals import K_ESCAPE, K_LEFT, K_RIGHT, KEYDOWN, K_m, K_r

from bbi.environments import GoRight
from bbi.models import BBI, ExpectationModel, SamplingModel

# List of models to cycle through
MODEL_CLASSES = [GoRight, SamplingModel, ExpectationModel, BBI]
MODEL_NAMES = ["GoRight", "SamplingModel", "ExpectationModel", "BBI"]
current_model_index = 0
env = MODEL_CLASSES[current_model_index]()
obs, info = env.reset()

pygame.init()
font = pygame.font.SysFont("Arial", 20)
running = True

# def render_model_info(screen, width, height):
#     """Render the current model info at bottom center."""
#     text_str = f"Current Model: {MODEL_NAMES[current_model_index]} (Press 'm' to change)"
#     text_surf = font.render(text_str, True, (0,0,0))
#     text_rect = text_surf.get_rect(center=(width//2, height - 15))
#     screen.blit(text_surf, text_rect)

while running:
    # Render the environment
    env.render()

    # Get the current pygame display surface
    screen = pygame.display.get_surface()
    if screen is None:
        # If env.render doesn't set up a display, do it here
        screen = pygame.display.set_mode((800, 600))
    width, height = screen.get_size()

    # Render model info (no button)
    # render_model_info(screen, width, height)
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
                # Step left action
                env.step(0)
            elif event.key == K_RIGHT:
                # Step right action
                env.step(1)
            elif event.key == K_m:
                # Change model
                current_model_index = (current_model_index + 1) % len(MODEL_CLASSES)
                env = MODEL_CLASSES[current_model_index]()
                env.reset()
            elif event.key == K_r:
                # Change model
                env.reset()

    # pygame.time.delay(100)
