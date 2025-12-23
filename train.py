import numpy as np
import torch
from environment import Environment
from networks import Network
import torch.nn.functional as F
import time

@torch.no_grad
def create_network(input_size: int, action_size: int, device: str):
  network = Network(input_size, action_size).to(device)
  return network

@torch.no_grad
def fitness(env: Environment, pop: list[Network], device='cpu'):
  scores = np.zeros(len(pop))
  it = 0
  max_it = 2000
  temp = 1.0
  while np.sum(env.get_done()) == 0 or it < max_it:
    states = env.get_states()
    states = torch.from_numpy(states).to(device)
    logits = [network.forward(states[i]) for i, network in enumerate(pop)]
    logits = torch.stack(logits, dim=0)
    logits = logits / temp
    actions = torch.multinomial(F.softmax(logits, dim=1), num_samples=1)[:, 0]
    actions = actions.cpu().numpy()
    # actions = np.ones(len(pop), dtype=np.int32)
    rewards = env.step(actions)
    scores += rewards
    it += 1
  return scores

@torch.no_grad
def train():
  device = 'cpu'
  env = Environment(10)
  pop = [create_network(env.state_size, env.action_size, device) for _ in range(10)]
  start = time.time()
  scores = fitness(env, pop, device)
  end = time.time()
  print(end - start)
  # print(scores)

if __name__ == '__main__':
  train()
