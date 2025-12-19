import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self, hidden_size: int, action_size: int):
    super().__init__()
    self.policy_fc1 = nn.Linear(hidden_size, 8)
    self.policy_fc2 = nn.Linear(8, action_size)
  
  def forward(self, state: torch.Tensor) -> torch.Tensor:
    p = self.policy_fc1(state)
    p = F.elu(p)
    p = self.policy_fc2(p)
    return p

  @staticmethod
  def save(model: 'Network', path: str, metadata={}):
    try:
      torch.save({
        'state_dict': model.state_dict(),
        'metadata': metadata
      }, path)
    except Exception as e:
      print(f"Error saving model: {e}")

  @staticmethod
  def load(path: str, default_metadata={}):
    try:
      checkpoint = torch.load(path)
      model = Network(checkpoint['metadata']['hidden_size'], checkpoint['metadata']['action_size'])
      model.load_state_dict(checkpoint['state_dict'])
      return model
    except Exception as e:
      print(f"Error loading model: {e}")
      print("Making new model...")
      return Network(default_metadata['hidden_size'], default_metadata['action_size'])
