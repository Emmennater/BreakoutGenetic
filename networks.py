import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self, input_size: int, action_size: int):
    super().__init__()
    self.policy_fc1 = nn.Linear(input_size, 8)
    self.policy_fc2 = nn.Linear(8, action_size)
    self.args = {'input_size': input_size, 'action_size': action_size}
  
  @torch.no_grad
  def forward(self, state: torch.Tensor) -> torch.Tensor:
    p = self.policy_fc1(state)
    p = F.elu(p)
    p = self.policy_fc2(p)
    return p

  @torch.no_grad
  @staticmethod
  def mutate(model: 'Network', freq: float, delta: float):
    params = list(model.parameters())
    for _, param in enumerate(params):
      mask = torch.rand_like(param) < freq
      param.add_(mask * torch.randn_like(param) * delta)

  @torch.no_grad
  @staticmethod
  def crossover(model1: 'Network', model2: 'Network', nc=2):
    """Simulated Binary Crossover"""
    child1 = copy.deepcopy(model1)
    child2 = copy.deepcopy(model2)

    params1 = list(model1.parameters())
    device = params1[0].device
    u = torch.rand(len(params1), device=device)

    beta_a = (2 * u) ** (1 / (nc + 1))
    beta_b = (1 / (2 * (1 - u))) ** (1 / (nc + 1))

    for i, (child1_param, child2_param) in enumerate(zip(child1.parameters(), child2.parameters())):
      beta = beta_a[i] if u[i] <= 0.5 else beta_b[i]
      x1 = child1_param.clone()
      x2 = child2_param.clone()
      child1_param.copy_(0.5 * ((1 + beta) * x1 + (1 - beta) * x2))
      child2_param.copy_(0.5 * ((1 - beta) * x1 + (1 + beta) * x2))

    return child1, child2

  @staticmethod
  def save(model: 'Network', path: str):
    try:
      torch.save({
        'state_dict': model.state_dict(),
        'args': model.args
      }, path)
    except Exception as e:
      print(f"Error saving model: {e}")

  @staticmethod
  def load(path: str, default_args={}):
    try:
      checkpoint = torch.load(path)
      model = Network(checkpoint['args']['input_size'], checkpoint['args']['action_size'])
      model.load_state_dict(checkpoint['state_dict'])
      return model
    except Exception as e:
      print(f"Error loading model: {e}")
      print("Making new model...")
      return Network(default_args['input_size'], default_args['action_size'])
