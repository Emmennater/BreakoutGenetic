from multiprocessing import freeze_support, set_start_method
from typing import Callable
from breakout import Breakout, render
from environment import Environment
from game import Game
from train import train
import pygame
import sys

def show(game: Game):
  screen_width, screen_height = 600, 700
  pygame.init()
  screen = pygame.display.set_mode((screen_width, screen_height))
  clock = pygame.time.Clock()
  running = True

  while running:
    clock.tick(60)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    render(game.environment, screen, screen_width, screen_height)
    pygame.display.flip()

  pygame.quit()

def live(game: Game, get_action: Callable):
  screen_width, screen_height = 600, 700
  pygame.init()
  screen = pygame.display.set_mode((screen_width, screen_height))
  clock = pygame.time.Clock()
  running = True
  total_reward = 0

  while running:
    clock.tick(60)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    # Get action
    action = get_action()

    # Update state
    if not game.terminal():
      _, reward = game.step(action)
      total_reward += reward
    else:
      game.reset()
      total_reward = 0

    render(game.environment, screen, screen_width, screen_height)
    
    # Total reward text
    font = pygame.font.Font(None, 36)
    text = font.render(f"Total reward: {total_reward:.2f}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()

  pygame.quit()

def play(Env: type[Environment]):
  game = Game(Env)
  game.environment.frame_skip = 1
  
  def get_action():
    action = 2
    if pygame.key.get_pressed()[pygame.K_LEFT]:
      action = 0
    if pygame.key.get_pressed()[pygame.K_RIGHT]:
      if action == 0:
        action = 2
      else:
        action = 1
    return action

  live(game, get_action)

if __name__ == "__main__":
  freeze_support()

  try:
    set_start_method('spawn', force=True)
  except RuntimeError:
    pass

  if '--play' in sys.argv:
    play(Breakout)
  else:
    train(Breakout)
