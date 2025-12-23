#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <net.h>
#include <random>
#include <iostream>
namespace py = pybind11;

mt19937 rng(0);
uniform_real_distribution<float> dist(0.0f, 1.0f);  

bool loaded = false;
Genome g;

void load(const std::string& filename) {
  loadGenome(g, filename);
  loaded = true;
}

int stepGenome(py::array_t<float> state) {
  if (!loaded) return 1;

  float random = dist(rng);
  auto state_mut = state.mutable_data();
  float reward = 0;
  int done = 0;
  
  State s;
  
  s.paddle_x = state_mut[0];
  s.ball_x = state_mut[1];
  s.ball_y = state_mut[2];
  s.ball_vx = state_mut[3];
  s.ball_vy = state_mut[4];
  
  for (int i = 0; i < BRICK_ROWS * BRICK_COLUMNS; i++) {
    s.bricks[i] = state_mut[5 + i];
  }

  int action = forward(g, s);

  envStep(&s, &action, &reward, &done, random);

  state_mut[0] = s.paddle_x;
  state_mut[1] = s.ball_x;
  state_mut[2] = s.ball_y;
  state_mut[3] = s.ball_vx;
  state_mut[4] = s.ball_vy;
  
  for (int i = 0; i < BRICK_ROWS * BRICK_COLUMNS; i++) {
    state_mut[5 + i] = s.bricks[i];
  }

  return done;
}

int step(py::array_t<float> state, int action) {
  mt19937 rng(0);
  uniform_real_distribution<float> dist(0.0f, 1.0f);
  float random = dist(rng);

  auto state_mut = state.mutable_data();
  float reward = 0;
  int done = 0;
  
  State s;
  
  s.paddle_x = state_mut[0];
  s.ball_x = state_mut[1];
  s.ball_y = state_mut[2];
  s.ball_vx = state_mut[3];
  s.ball_vy = state_mut[4];
  
  for (int i = 0; i < BRICK_ROWS * BRICK_COLUMNS; i++) {
    s.bricks[i] = state_mut[5 + i];
  }

  envStep(&s, &action, &reward, &done, random);

  state_mut[0] = s.paddle_x;
  state_mut[1] = s.ball_x;
  state_mut[2] = s.ball_y;
  state_mut[3] = s.ball_vx;
  state_mut[4] = s.ball_vy;
  
  for (int i = 0; i < BRICK_ROWS * BRICK_COLUMNS; i++) {
    state_mut[5 + i] = s.bricks[i];
  }

  return done;
}

PYBIND11_MODULE(link, m) {
  m.def("load", &load);
  m.def("stepGenome", &stepGenome);
  m.def("step", &step);
}
