import numpy as np
import env

class Environment:
  STATE_SIZE = 1

  def __init__(self, max_agents: int, seed: int = -1, device: str = 'cpu'):
    """
    Initialize the environment.

    :param max_agents: The maximum number of agents in the environment.
    :param seed: The seed for the random number generator.
    :param device: The device to use for the environment.
    """
    self.max_agents = max_agents
    self.n_agents = max_agents
    self.device = device
    self.state_size = env.get_state_size()
    self.reset()
    env.init(self.n_agents, seed)

  def reset(self):
    """
    Reset the environment.
    """
    self.states = np.zeros((self.n_agents, self.state_size), dtype=np.float32)
    self.done = np.zeros(self.n_agents, dtype=np.bool)
    self.rewards = np.zeros(self.n_agents, dtype=np.float32)
    env.reset(self.states)

  def step(self, actions: np.ndarray):
    """
    Perform a step in the environment.

    :param actions: A numpy array of actions for each agent with shape (n_agents,)
    """
    env.step(actions, self.states, self.done, self.rewards)

  def get_states(self) -> np.ndarray:
    """
    Get the current states of the environment.

    :return: A numpy array of states with shape (n_agents, state_size)
    """
    return self.states

  def get_done(self) -> np.ndarray:
    """
    Get the done flags for each agent.

    :return: A numpy array of done flags with shape (n_agents,)
    """
    return self.done
  
  def get_rewards(self) -> np.ndarray:
    """
    Get the rewards for each agent.

    :return: A numpy array of rewards with shape (n_agents,)
    """
    return self.rewards

  def get_state_size(self) -> int:
    """
    Get the state size of the environment.
    """
    return self.state_size

  def get_n_agents(self) -> int:
    """
    Get the number of agents in the environment.
    """
    return self.n_agents
