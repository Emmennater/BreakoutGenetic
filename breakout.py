import random
import pygame
import torch
from environment import Environment
import numpy as np

# Game parameters
BRICK_ROWS = 7
BRICK_COLUMNS = 10
STATE_SIZE = 5
PADDLE_SPEED = 0.01
BALL_SPEED = 0.01
HALF_PADDLE_WIDTH = 0.075
HALF_PADDLE_HEIGHT = 0.01
PADDLE_Y = 0.95
BALL_RADIUS = 0.015
BRICK_TOP = 0.1
BRICK_BOTTOM = 0.3

class Breakout(Environment):
  def __init__(self):
    super().__init__()

    self.is_done = False
    self.paddle_x = 0.5
    self.ball_x = 0.5
    self.ball_y = 0.5
    self.ball_vx = 0
    self.ball_vy = BALL_SPEED
    self.bricks = [1] * BRICK_ROWS * BRICK_COLUMNS

  def terminal(self):
    return self.is_done

  def pos_to_brick_idx(self, x, y):
    """Map a position in the world to the index of the corresponding brick."""
    if y < BRICK_TOP or y >= BRICK_BOTTOM or x < 0 or x >= 1:
      return None
    
    row = np.floor((y - BRICK_TOP) / (BRICK_BOTTOM - BRICK_TOP) * BRICK_ROWS)
    col = np.floor(x * BRICK_COLUMNS)

    return int(row * BRICK_COLUMNS + col)

  def step(self, action: int):
    reward = 0.001

    # Move paddle
    if action == 0:
      self.paddle_x -= PADDLE_SPEED
    elif action == 1:
      self.paddle_x += PADDLE_SPEED

    # Constrain paddle
    if self.paddle_x < HALF_PADDLE_WIDTH:
      self.paddle_x = HALF_PADDLE_WIDTH
    if self.paddle_x > 1 - HALF_PADDLE_WIDTH:
      self.paddle_x = 1 - HALF_PADDLE_WIDTH

    # Move ball
    self.ball_x += self.ball_vx
    self.ball_y += self.ball_vy

    # Constrain ball vertically
    if self.ball_y < BALL_RADIUS:
      self.ball_y = BALL_RADIUS
      self.ball_vy = -self.ball_vy
    elif self.ball_y > 1 - BALL_RADIUS:
      self.ball_y = 1 - BALL_RADIUS
      self.ball_vy = -self.ball_vy
      reward -= 1
      self.is_done = True

    # Constrain ball horizontally
    if self.ball_x < BALL_RADIUS:
      self.ball_x = BALL_RADIUS
      self.ball_vx = -self.ball_vx
    elif self.ball_x > 1 - BALL_RADIUS:
      self.ball_x = 1 - BALL_RADIUS
      self.ball_vx = -self.ball_vx

    ball_top = self.ball_y - BALL_RADIUS
    ball_bottom = self.ball_y + BALL_RADIUS
    ball_left = self.ball_x - BALL_RADIUS
    ball_right = self.ball_x + BALL_RADIUS

    # Ball colliding with paddle
    if self.ball_vy > 0:
      paddle_left = self.paddle_x - HALF_PADDLE_WIDTH
      paddle_right = self.paddle_x + HALF_PADDLE_WIDTH
      paddle_top = PADDLE_Y - HALF_PADDLE_HEIGHT
      paddle_bottom = PADDLE_Y + HALF_PADDLE_HEIGHT

      if (ball_right >= paddle_left and
        ball_left <= paddle_right and
        ball_bottom >= paddle_top and
        ball_top <= paddle_bottom):
        self.ball_y = PADDLE_Y - BALL_RADIUS
        self.ball_vy = -self.ball_vy
        ball_to_paddle = (self.ball_x - self.paddle_x) / HALF_PADDLE_WIDTH + random.random() * 0.2 - 0.1
        self.ball_vx += ball_to_paddle * BALL_SPEED * 0.5
        ball_speed = np.sqrt(self.ball_vx ** 2 + self.ball_vy ** 2)
        self.ball_vx = self.ball_vx / ball_speed * BALL_SPEED
        self.ball_vy = self.ball_vy / ball_speed * BALL_SPEED
        reward += 1

    # Ball colliding with bricks
    brick_tl = self.pos_to_brick_idx(ball_left, ball_top)
    brick_tr = self.pos_to_brick_idx(ball_right, ball_top)
    brick_bl = self.pos_to_brick_idx(ball_left, ball_bottom)
    brick_br = self.pos_to_brick_idx(ball_right, ball_bottom)

    if brick_tl is not None and self.bricks[brick_tl] == 1:
      reward += 1.0
      self.bricks[brick_tl] = 0
      self.ball_vy = -self.ball_vy
    elif brick_tr is not None and self.bricks[brick_tr] == 1:
      reward += 1.0
      self.bricks[brick_tr] = 0
      self.ball_vy = -self.ball_vy
    elif brick_bl is not None and self.bricks[brick_bl] == 1:
      reward += 1.0
      self.bricks[brick_bl] = 0
      self.ball_vy = -self.ball_vy
    elif brick_br is not None and self.bricks[brick_br] == 1:
      reward += 1.0
      self.bricks[brick_br] = 0
      self.ball_vy = -self.ball_vy

    reward = np.clip(reward, -1, 1)

    return self.get_state(), reward

  def get_state(self) -> torch.Tensor:
    state = torch.zeros(STATE_SIZE, dtype=torch.float32)
    state[0] = self.paddle_x
    state[1] = self.ball_x
    state[2] = self.ball_y
    state[3] = self.ball_vx
    state[4] = self.ball_vy
    # state[6:6+BRICK_ROWS*BRICK_COLUMNS] = torch.tensor(self.bricks)
    return state

  def get_actions(self) -> list[int]:
    return [0, 1, 2]

  @staticmethod
  def config() -> dict:
    return {
      'n_actors': 1,
      'frame_skip': 4,
      'hidden_size': 8,
      'mutate_rate': 0.05,
      'mutate_std': 0.025
    }

def render(env: Breakout, screen: pygame.Surface, screen_width, screen_height):
  brick_width = screen_width // BRICK_COLUMNS
  brick_height = screen_height * (BRICK_BOTTOM - BRICK_TOP) // BRICK_ROWS

  PADDLE_COLOR = (107, 115, 117)
  BALL_COLOR = (250, 255, 255)
  RAINBOW = [
    (255, 85, 85),      # red
    (255, 120, 70),    # orange
    (255, 190, 20),    # yellow
    (90, 230, 90),      # green
    (70, 80, 255),    # blue
    (130, 80, 255),    # purple
    (0, 240, 240)   #  light blue
  ]

  screen.fill((12, 23, 31))

  # Bricks
  for i in range(0, BRICK_COLUMNS * BRICK_ROWS):
    if env.bricks[i] == 0: continue
    col = i % BRICK_COLUMNS
    row = i // BRICK_COLUMNS
    left = col * brick_width + 1
    top = BRICK_TOP * screen_height + row * brick_height + 1
    width = brick_width - 2
    height = brick_height - 2
    rect = pygame.Rect(left, top, width, height)
    pygame.draw.rect(screen, RAINBOW[row], rect)

  # Paddle
  paddle_width = HALF_PADDLE_WIDTH * screen_width * 2
  paddle_height = HALF_PADDLE_HEIGHT * screen_height * 2
  paddle_left = env.paddle_x * screen_width - paddle_width / 2
  paddle_top = PADDLE_Y * screen_height - paddle_height / 2
  paddle = pygame.Rect(paddle_left, paddle_top, paddle_width, paddle_height)
  pygame.draw.rect(screen, PADDLE_COLOR, paddle)

  # Ball
  ball_x = env.ball_x * screen_width
  ball_y = env.ball_y * screen_height
  ball_r = BALL_RADIUS * screen_height
  pygame.draw.circle(screen, BALL_COLOR, (ball_x, ball_y), ball_r)
