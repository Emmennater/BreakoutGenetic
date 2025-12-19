import timeit
import torch
import torch.nn as nn
import env

class Net(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x):
    return torch.tensor([0.0, 1.0])

n_agents = 256
env.init(n_agents)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Allocate once on CPU
actions = torch.zeros(n_agents, dtype=torch.int32, device='cpu')
observations = torch.zeros(n_agents, dtype=torch.float32, device='cpu')
done_flags = torch.zeros(n_agents, dtype=torch.bool, device='cpu')

net = Net()

def compute():
  for t in range(100):
    # Example: batched PyTorch NN (on GPU)
    # Convert observations to GPU for NN
    states_gpu = observations.to(device)
    actions_gpu = net(states_gpu).argmax(dim=0)
    actions[:] = actions_gpu.to('cpu')  # copy once to CPU tensor

    # Step environment (zero-copy, multithreaded)
    env.step(actions.numpy(), observations.numpy(), done_flags.numpy())

    # Reset done agents
    done_indices = done_flags.nonzero(as_tuple=True)[0]
    observations[done_indices] = 0.0
    done_flags[done_indices] = False


t1 = timeit.default_timer()
compute()
t2 = timeit.default_timer()
print(f"Time: {t2 - t1} seconds")
