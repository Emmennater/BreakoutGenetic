"""
A simple breakout environment.
"""

import numpy as np

def init(n_agents: int, seed: int) -> None:
  """
  Initialize the environment for `n_agents` agents.

  :param n_agents: The number of agents in the environment.
  :param seed: The seed for the random number generator.
  """

def step(actions: np.ndarray, state_out: np.ndarray, done_out: np.ndarray, reward_out: np.ndarray) -> None:
  """
  Perform a step in the environment.

  :param actions: A tensor of actions for each agent.
  :param state_out: A tensor to store the states for each agent.
  :param done_out: A tensor to store the done flags for each agent.
  """

def reset(state_out: np.ndarray) -> None:
  """
  Reset the environment.

  :param state_out: A tensor to store the states for each agent.
  """

def get_state_size() -> int:
  """
  Get the state size of the environment.
  """

def get_action_size() -> int:
  """
  Get the action size of the environment.
  """
