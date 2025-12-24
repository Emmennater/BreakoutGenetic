#pragma once
#include <bitset>
#include <vector>
using namespace std;

namespace EnvConfig {
  constexpr int BRICK_ROWS = 7;
  constexpr int BRICK_COLUMNS = 10;
  constexpr float PADDLE_SPEED = 0.005f;
  constexpr float BALL_SPEED = 0.01f;
  constexpr float HALF_PADDLE_WIDTH = 0.075f;
  constexpr float HALF_PADDLE_HEIGHT = 0.01f;
  constexpr float PADDLE_Y = 0.95f;
  constexpr float BALL_RADIUS = 0.015f;
  constexpr float BRICK_TOP = 0.1f;
  constexpr float BRICK_BOTTOM = 0.3f;
  constexpr int STATE_SIZE = 5 + BRICK_ROWS * BRICK_COLUMNS;
  constexpr int ACTION_SIZE = 3;
}

using namespace EnvConfig;

struct State {
  float paddle_x = 0.5;
  float ball_x = 0.5;
  float ball_y = 0.5;
  float ball_vx = 0.0;
  float ball_vy = BALL_SPEED;
  float bricks[BRICK_ROWS * BRICK_COLUMNS];

  State() {
    for (int i = 0; i < BRICK_ROWS * BRICK_COLUMNS; i++) {
      bricks[i] = 1.0f;
    }
  }
};

void envStep(State* state, int* action, float* reward, int* done, float rng);
