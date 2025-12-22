import numpy as np
from environment import Environment

class Games:
  def __init__(self, environment: Environment):
    """
    Initialize the environment.
    
    :param environment: The environment to use.
    """

    self.environment = environment
    self.frame_skip = 1
    self.n_agents = environment.get_n_agents()
    self.state_size = environment.get_state_size()

  def step(self, actions: np.ndarray) -> np.ndarray:
    """
    Take a step in the environment.

    :param actions: A numpy array of actions for each agent of shape (n_agents,).
    :return: A numpy array of rewards for each agent of shape (n_agents,).
    """

    total_reward = np.zeros(shape=(self.n_agents,), dtype=np.float32)
    for _ in range(self.frame_skip):
      self.environment.step(actions)
      total_reward += self.environment.get_rewards()
    return total_reward.copy()

  def reset(self):
    """
    Reset the environment.
    """

    self.environment.reset()

  def get_done(self):
    """
    Get the done flags for each agent.
    
    :return: A numpy array of done flags for each agent of shape (n_agents,).
    """

    return self.environment.get_done().copy()
  
  def get_states(self):
    """
    Get the current states of the environment.
    
    :return: A numpy array of states for each agent of shape (n_agents, state_size).
    """

    return self.environment.get_states().copy()
