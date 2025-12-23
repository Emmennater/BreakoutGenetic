#include <bitset>
#include <cmath>
#include <env.h>
#include <vector>

int pos2idx(float y, float x) {
  if (y < BRICK_TOP || y >= BRICK_BOTTOM || x < 0 || x >= 1) {
    return -1;
  }
  
  int row = static_cast<int>((y - BRICK_TOP) / (BRICK_BOTTOM - BRICK_TOP) * BRICK_ROWS);
  int col = static_cast<int>(x * BRICK_COLUMNS);

  return row * BRICK_COLUMNS + col;
}

void envStep(State* state, int* action, float* reward, int* done, float random) {
  // *reward += 0.01f;

  // Movement
  state->paddle_x += static_cast<float>(*action - 1) * PADDLE_SPEED;
  state->paddle_x = std::max(HALF_PADDLE_WIDTH, std::min(1.0f - HALF_PADDLE_WIDTH, state->paddle_x));
  state->ball_x += state->ball_vx;
  state->ball_y += state->ball_vy;
  
  // Ball vs wall
  if (state->ball_x < BALL_RADIUS) {
    state->ball_x = BALL_RADIUS;
    state->ball_vx = -state->ball_vx;
  } else if (state->ball_x > 1 - BALL_RADIUS) {
    state->ball_x = 1 - BALL_RADIUS;
    state->ball_vx = -state->ball_vx;
  }
  if (state->ball_y < BALL_RADIUS) {
    state->ball_y = BALL_RADIUS;
    state->ball_vy = -state->ball_vy;
  } else if (state->ball_y > 1 - BALL_RADIUS) {
    state->ball_y = 1 - BALL_RADIUS;
    state->ball_vy = -state->ball_vy;
    *done = 1;
  }

  // Ball vs paddle
  if (state->ball_y + BALL_RADIUS > PADDLE_Y - HALF_PADDLE_HEIGHT &&
      state->ball_y - BALL_RADIUS < PADDLE_Y + HALF_PADDLE_HEIGHT &&
      state->ball_x + BALL_RADIUS > state->paddle_x - HALF_PADDLE_WIDTH &&
      state->ball_x - BALL_RADIUS < state->paddle_x + HALF_PADDLE_WIDTH &&
      state->ball_vy > 0.0f) {
    state->ball_y = PADDLE_Y - HALF_PADDLE_HEIGHT;
    state->ball_vy = -state->ball_vy;
    float ball2paddle = state->ball_x - state->paddle_x;
    float perturb = (ball2paddle / HALF_PADDLE_WIDTH + random * 0.2f - 0.1f) * BALL_SPEED * 0.5;
    state->ball_vx += perturb;
    float ball_speed = std::sqrt(state->ball_vx * state->ball_vx + state->ball_vy * state->ball_vy);
    state->ball_vx *= BALL_SPEED / ball_speed;
    state->ball_vy *= BALL_SPEED / ball_speed;
  }

  // Ball vs bricks
  float ball_t = state->ball_y - BALL_RADIUS;
  float ball_b = state->ball_y + BALL_RADIUS;
  float ball_r = state->ball_x + BALL_RADIUS;
  float ball_l = state->ball_x - BALL_RADIUS;
  int brick_tr = pos2idx(ball_t, ball_r);
  int brick_tl = pos2idx(ball_t, ball_l);
  int brick_br = pos2idx(ball_b, ball_r);
  int brick_bl = pos2idx(ball_b, ball_l);

  if (brick_tr != -1 && state->bricks[brick_tr]) {
    state->bricks[brick_tr] = 0;
    state->ball_vy = -state->ball_vy;
    *reward += 1.0f;
  } else if (brick_tl != -1 && state->bricks[brick_tl]) {
    state->bricks[brick_tl] = 0;
    state->ball_vy = -state->ball_vy;
    *reward += 1.0f;
  } else if (brick_br != -1 && state->bricks[brick_br]) {
    state->bricks[brick_br] = 0;
    state->ball_vy = -state->ball_vy;
    *reward += 1.0f;
  } else if (brick_bl != -1 && state->bricks[brick_bl]) {
    state->bricks[brick_bl] = 0;
    state->ball_vy = -state->ball_vy;
    *reward += 1.0f;
  }
}
