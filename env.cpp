#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <thread>
#include <atomic>
#include <omp.h>
#include <random>
#include <bitset>
#include <iostream>

#define BRICK_ROWS 7
#define BRICK_COLUMNS 10
#define PADDLE_SPEED 0.01f
#define BALL_SPEED 0.01f
#define HALF_PADDLE_WIDTH 0.075f
#define HALF_PADDLE_HEIGHT 0.01f
#define PADDLE_Y 0.95f
#define BALL_RADIUS 0.015f
#define BRICK_TOP 0.1f
#define BRICK_BOTTOM 0.3f

namespace py = pybind11;

int get_state_size() {
  return 5 + BRICK_ROWS * BRICK_COLUMNS;
}

int get_action_size() {
  return 3;
}

typedef struct State {
  float paddle_x = 0.5f;
  float ball_x = 0.5f;
  float ball_y = 0.5f;
  float ball_vx = 0.0f;
  float ball_vy = BALL_SPEED;
  std::bitset<BRICK_ROWS * BRICK_COLUMNS> bricks = std::bitset<BRICK_ROWS * BRICK_COLUMNS>().set();
} State;

inline int pos2idx(float y, float x) {
  if (y < BRICK_TOP || y >= BRICK_BOTTOM || x < 0 || x >= 1) {
    return -1;
  }
  
  int row = static_cast<int>((y - BRICK_TOP) / (BRICK_BOTTOM - BRICK_TOP) * BRICK_ROWS);
  int col = static_cast<int>(x * BRICK_COLUMNS);

  return row * BRICK_COLUMNS + col;
}

void state_step(State* state, int action, bool* done, float* reward, float rng, int* removed) {
  *reward += 0.01f;

  // Movement
  state->paddle_x += static_cast<float>(action - 1) * PADDLE_SPEED;
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
    *done = true;
  }

  // Ball vs paddle
  if (state->ball_y + BALL_RADIUS > PADDLE_Y - HALF_PADDLE_HEIGHT &&
      state->ball_y - BALL_RADIUS < PADDLE_Y + HALF_PADDLE_HEIGHT &&
      state->ball_x + BALL_RADIUS > state->paddle_x - HALF_PADDLE_WIDTH &&
      state->ball_x - BALL_RADIUS < state->paddle_x + HALF_PADDLE_WIDTH &&
      state->ball_vy > 0.0f) {
    *reward += 1.0f;
    state->ball_y = PADDLE_Y - HALF_PADDLE_HEIGHT;
    state->ball_vy = -state->ball_vy;
    float ball2paddle = state->ball_x - state->paddle_x;
    float perturb = (ball2paddle / HALF_PADDLE_WIDTH + rng * 0.2f - 0.1f) * BALL_SPEED * 0.5;
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
    *removed = brick_tr;
  } else if (brick_tl != -1 && state->bricks[brick_tl]) {
    state->bricks[brick_tl] = 0;
    state->ball_vy = -state->ball_vy;
    *reward += 1.0f;
    *removed = brick_tl;
  } else if (brick_br != -1 && state->bricks[brick_br]) {
    state->bricks[brick_br] = 0;
    state->ball_vy = -state->ball_vy;
    *reward += 1.0f;
    *removed = brick_br;
  } else if (brick_bl != -1 && state->bricks[brick_bl]) {
    state->bricks[brick_bl] = 0;
    state->ball_vy = -state->ball_vy;
    *reward += 1.0f;
    *removed = brick_bl;
  }
}

bool g_is_reset = false;
int g_n_agents;
std::vector<State> g_state;
std::vector<uint8_t> g_done;

void reset(py::array_t<float> state_out) {
  auto obs_buf = state_out.mutable_unchecked<2>();
  g_is_reset = true;
  for (int i = 0; i < g_n_agents; i++) {
    g_state[i] = State();
    g_done[i] = false;
    obs_buf(i, 0) = g_state[i].paddle_x;
    obs_buf(i, 1) = g_state[i].ball_x;
    obs_buf(i, 2) = g_state[i].ball_y;
    obs_buf(i, 3) = g_state[i].ball_vx;
    obs_buf(i, 4) = g_state[i].ball_vy;
    for (int j = 0; j < BRICK_ROWS * BRICK_COLUMNS; j++) {
      obs_buf(i, 5 + j) = g_state[i].bricks[j];
    }
  }
}

void init(int n_agents, int seed) {
  if (!g_is_reset) throw std::runtime_error("Call reset() before init()");
  g_n_agents = n_agents;
  if (seed != -1) srand(seed);
  g_state.resize(n_agents);
  g_done.resize(n_agents);
}

void step(
  py::array_t<int> actions,
  py::array_t<float> state_out,
  py::array_t<bool> done_out,
  py::array_t<float> reward_out
) {
  omp_set_num_threads(4);

  // Release GIL for multithreading
  py::gil_scoped_release release;

  auto done_ptr = done_out.mutable_data();
  auto reward_ptr = reward_out.mutable_data();
  auto obs_buf = state_out.mutable_unchecked<2>();

  float rng = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

  #pragma omp parallel for
  for (int i = 0; i < g_n_agents; i++) {
    if (!g_done[i]) {
      int removed = -1;
      state_step(&g_state[i], actions.at(i), &done_ptr[i], &reward_ptr[i], rng, &removed);
      obs_buf(i, 0) = g_state[i].paddle_x;
      obs_buf(i, 1) = g_state[i].ball_x;
      obs_buf(i, 2) = g_state[i].ball_y;
      obs_buf(i, 3) = g_state[i].ball_vx;
      obs_buf(i, 4) = g_state[i].ball_vy;
      if (removed != -1) obs_buf(i, removed + 5) = 0.0f;
      g_done[i] = done_ptr[i];
    }
  }
}

PYBIND11_MODULE(env, m) {
  m.doc() = "Environment";
  m.def("init", &init);
  m.def("step", &step);
  m.def("reset", &reset);
  m.def("get_state_size", &get_state_size);
  m.def("get_action_size", &get_action_size);
}
