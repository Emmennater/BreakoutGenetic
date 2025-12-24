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

int stepGenome(py::array_t<float> state, py::array_t<float> reward) {
  if (!loaded) return 1;

  float random = dist(rng);
  auto state_mut = state.mutable_data();
  auto reward_mut = reward.mutable_data();
  int done = 0;
  State s;
  
  memcpy(&s, state_mut, sizeof(State));
  int action = forward(g, s);
  envStep(&s, &action, &reward_mut[0], &done, random);
  memcpy(state_mut, &s, sizeof(State));

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
  
  memcpy(&s, state_mut, sizeof(State));
  envStep(&s, &action, &reward, &done, random);
  memcpy(state_mut, &s, sizeof(State));

  return done;
}

PYBIND11_MODULE(link, m) {
  m.def("load", &load);
  m.def("stepGenome", &stepGenome);
  m.def("step", &step);
}
