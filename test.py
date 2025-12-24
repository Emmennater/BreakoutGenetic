import random
import sys
import link
import numpy as np
import pygame

def get_state():
  state = np.zeros((5 + 7 * 10), dtype=np.float32)
  # ball_angle = random.uniform(np.pi * 0.25, np.pi * 0.25)
  # ball_speed = 0.01
  # state[0] = 0.5
  # state[1] = random.uniform(0.4, 0.5)
  # state[2] = random.uniform(0.2, 0.8)
  # state[3] = np.cos(ball_angle) * ball_speed
  # state[4] = np.sin(ball_angle) * ball_speed
  state[0] = 0.5
  state[1] = 0.5
  state[2] = 0.5
  state[3] = 0.0
  state[4] = 0.01
  state[5:] = 1
  return state

def render(state: np.ndarray, screen: pygame.Surface, screen_width, screen_height):
  BRICK_ROWS = 7
  BRICK_COLUMNS = 10
  HALF_PADDLE_WIDTH = 0.075
  HALF_PADDLE_HEIGHT = 0.01
  PADDLE_Y = 0.95
  BALL_RADIUS = 0.015
  BRICK_TOP = 0.1
  BRICK_BOTTOM = 0.3

  paddle_x = state[0].item()
  ball_x = state[1].item()
  ball_y = state[2].item()
  bricks = state[5:]

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
    if bricks[i].item() == 0: continue
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
  paddle_left = paddle_x * screen_width - paddle_width / 2
  paddle_top = PADDLE_Y * screen_height - paddle_height / 2
  paddle = pygame.Rect(paddle_left, paddle_top, paddle_width, paddle_height)
  pygame.draw.rect(screen, PADDLE_COLOR, paddle)

  # Ball
  ball_x = ball_x * screen_width
  ball_y = ball_y * screen_height
  ball_r = BALL_RADIUS * screen_height
  pygame.draw.circle(screen, BALL_COLOR, (ball_x, ball_y), ball_r)

def test(ai=False):
  state = get_state()
  
  link.load("genomes/0.net")
  # link.load("best_models/best.net")
  
  pygame.init()
  screen_width, screen_height = 600, 700
  screen = pygame.display.set_mode((screen_width, screen_height))
  clock = pygame.time.Clock()
  running = True
  keys_down = {}
  frame = 0
  total_reward = 0
  reward = np.zeros(1, dtype=np.float32)

  while running:
    clock.tick(60)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.KEYDOWN:
        keys_down[event.key] = True
      if event.type == pygame.KEYUP:
        keys_down[event.key] = False

    if ai:
      done = link.stepGenome(state, reward)
      total_reward += reward[0]
      reward[0] = 0
    else:
      action = 1
      if pygame.K_LEFT in keys_down and keys_down[pygame.K_LEFT]:
        action -= 1
      if pygame.K_RIGHT in keys_down and keys_down[pygame.K_RIGHT]:
        action += 1
      done = link.step(state, action)

    render(state, screen, screen_width, screen_height)
    frame += 1
    
    txt = "Frame: %d" % frame
    font = pygame.font.Font(None, 24)
    text = font.render(txt, True, (255, 255, 255))
    screen.blit(text, (10, 10))

    txt = "Reward: %.0f" % total_reward
    font = pygame.font.Font(None, 24)
    text = font.render(txt, True, (255, 255, 255))
    screen.blit(text, (10, 40))

    if done:
      state = get_state()
      frame = 0
      total_reward = 0

    pygame.display.flip()

  pygame.quit()

if __name__ == "__main__":
  test('--play' not in sys.argv)
