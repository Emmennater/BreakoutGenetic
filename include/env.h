#pragma once
#include <bitset>
using namespace std;

class Environment {
  static const int BRICK_ROWS = 7;
  static const int BRICK_COLUMNS = 10;
  static constexpr float PADDLE_SPEED = 0.01f;
  static constexpr float BALL_SPEED = 0.01f;
  static constexpr float HALF_PADDLE_WIDTH = 0.075f;
  static constexpr float HALF_PADDLE_HEIGHT = 0.01f;
  static constexpr float PADDLE_Y = 0.95f;
  static constexpr float BALL_RADIUS = 0.015f;
  static constexpr float BRICK_TOP = 0.1f;
  static constexpr float BRICK_BOTTOM = 0.3f;

public:
  struct State {
    float paddle_x;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    bitset<BRICK_ROWS * BRICK_COLUMNS> bricks;
  };

  int pos2idx(float, float);
  void step(State*, int, float*, bool*, float);
};