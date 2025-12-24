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
    // *reward += 1.0f;
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
  int brick_idx = -1;

  if (brick_tr != -1 && state->bricks[brick_tr]) {
    brick_idx = brick_tr;
  } else if (brick_tl != -1 && state->bricks[brick_tl]) {
    brick_idx = brick_tl;
  } else if (brick_br != -1 && state->bricks[brick_br]) {
    brick_idx = brick_br;
  } else if (brick_bl != -1 && state->bricks[brick_bl]) {
    brick_idx = brick_bl;
  }

  if (brick_idx != -1) {
    // int col = brick_idx % BRICK_COLUMNS;
    // int row = brick_idx / BRICK_COLUMNS;
    
    // float w1 = state->ball_vx;
    // float h1 = state->ball_vy;
    // float w2 = 1.0f / BRICK_COLUMNS;
    // float h2 = (BRICK_BOTTOM - BRICK_TOP) / BRICK_ROWS;
    // float x0 = state->ball_x - state->ball_vx;
    // float y0 = state->ball_y - state->ball_vy;
    // float x1 = state->ball_x;
    // float y1 = state->ball_y;
    // float x2 = col * w2 + w2 * 0.5f;
    // float y2 = BRICK_TOP + row * h2 + h2 * 0.5f;
    
    // float xmin = x2 - w2/2 - w1/2;
    // float xmax = x2 + w2/2 - w1/2;
    // float ymin = y2 - h2/2 - h1/2;
    // float ymax = y2 + h2/2 + h1/2;
    // float dx = x1 - x0;
    // float dy = y1 - y0;
    // float tx1 = (xmin - x0) / dx;
    // float tx2 = (xmax - x0) / dx;
    // float ty1 = (ymin - y0) / dy;
    // float ty2 = (ymax - y0) / dy;
    // float txmin = min(tx1, tx2);
    // float tymin = min(ty1, ty2);
    
    // if (txmin > tymin) {
    //   state->ball_vx = -state->ball_vx;
    // } else {
    //   state->ball_vy = -state->ball_vy;
    // }

    state->ball_vy = -state->ball_vy;
    state->bricks[brick_idx] = 0;
    *reward += 1.0f;
    // *reward += BRICK_ROWS - row;
  }
}
