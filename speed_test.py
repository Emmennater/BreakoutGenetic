import numpy as np
from environment import Environment
import timeit

def compute_2(n_agents, n_steps):
  env = Environment(n_agents)
  actions = np.ones(n_agents, dtype=np.int32)
  for _ in range(n_steps):
    env.step(actions)

print(timeit.timeit("compute_2(100000, 1000)", setup="from __main__ import compute_2", number=1))
