// env.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <thread>
#include <atomic>
#include <omp.h>

namespace py = pybind11;

int g_n_agents;
std::vector<int> g_state;   // simplified representation
std::vector<bool> g_done;

void init(int n_agents) {
  g_n_agents = n_agents;
  g_state.resize(n_agents, 0);
  g_done.resize(n_agents, false);
}

void reset() {
  for (int i = 0; i < g_n_agents; i++) {
    g_state[i] = 0;
    g_done[i] = false;
  }
}

// actions: int array of size n_agents
// observation_out: int array of size n_agents
// done_out: bool array of size n_agents
void step(py::array_t<int> actions,
      py::array_t<int> observation_out,
      py::array_t<bool> done_out)
{
  // Release GIL for multithreading
  py::gil_scoped_release release;

  auto actions_ptr = actions.mutable_data();
  auto obs_ptr = observation_out.mutable_data();
  auto done_ptr = done_out.mutable_data();

  // Parallel step with OpenMP
  #pragma omp parallel for
  for (int i = 0; i < g_n_agents; i++) {
    if (!g_done[i]) {
      g_state[i] += actions_ptr[i];       // dummy simulation
      obs_ptr[i] = g_state[i];         // observation
      g_done[i] = (g_state[i] > 10);     // done condition
      done_ptr[i] = g_done[i];
    } else {
      obs_ptr[i] = g_state[i];
      done_ptr[i] = true;
    }
  }
}

PYBIND11_MODULE(env, m) {
  m.doc() = "Environment";
  m.def("init", &init, "Initialize environment", py::arg("n_agents"));
  m.def("step", &step, "Step environment", py::arg("actions"), py::arg("observation_out"), py::arg("done_out"));
  m.def("reset", &reset, "Reset environment");
}
