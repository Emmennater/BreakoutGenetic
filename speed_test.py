import torch
from breakout import Breakout
from game import Game
import env
import timeit

# 1 thousand steps per second
def compute_1(n_agents, n_steps):
  games = [Game(Breakout) for _ in range(n_agents)]
  for i in range(n_steps):
    for game in games:
      game.step(2)

# 77 million steps per second (77,000x speedup)
def compute_2(n_agents, n_steps):
  actions = torch.ones(n_agents, dtype=torch.int32, device='cpu') * 1
  observations = torch.zeros((n_agents, 5), dtype=torch.float32, device='cpu')
  done_flags = torch.zeros(n_agents, dtype=torch.bool, device='cpu')
  rewards = torch.zeros(n_agents, dtype=torch.float32, device='cpu')
  env.init(n_agents, -1)
  for _ in range(n_steps):
    env.step(actions.numpy(), observations.numpy(), done_flags.numpy(), rewards.numpy())

print(timeit.timeit("compute_1(100, 100)", setup="from __main__ import compute_1", number=1))
# print(timeit.timeit("compute_2(10000000, 1000)", setup="from __main__ import compute_2", number=1))
