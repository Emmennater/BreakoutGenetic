from pyparsing import abstractmethod
import torch

class Environment:
  def __init__(self):
    pass

  @abstractmethod
  def get_state(self) -> torch.Tensor:
    pass

  @abstractmethod
  def terminal(self) -> bool:
    pass

  @abstractmethod
  def reset(self):
    pass

  @abstractmethod
  def step(self, action: int) -> tuple[torch.Tensor, float]:
    pass

  @abstractmethod
  def get_actions(self) -> list[int]:
    pass

  @staticmethod
  @abstractmethod
  def config() -> dict:
    pass
