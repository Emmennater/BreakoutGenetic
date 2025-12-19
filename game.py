
from pyparsing import abstractmethod
import torch
from environment import Environment

class Game:
  def __init__(self, Env: type[Environment]):
    self.Env = Env
    self.environment = Env()
    self.frame_skip = 1

  def step(self, action: int) -> tuple[torch.Tensor, float]:
    total_reward = 0
    for _ in range(self.frame_skip):
      state, reward = self.environment.step(action)
      total_reward += reward
    return state, total_reward

  def reset(self):
    self.environment = self.Env()
    return self.environment.get_state()

  def terminal(self):
    return self.environment.terminal()
  
  def get_state(self):
    return self.environment.get_state()
