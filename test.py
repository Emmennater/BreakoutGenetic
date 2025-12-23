import numpy as np

I = 5
H = 8
O = 3

def load_genome(filename: str, I: int, H: int, O: int):
  """
  Load a genome (policy network) from a binary file.
  """
  genome_size = H * I + H + H * H + H + O * H + O
  genome = np.fromfile(filename, dtype=np.float32, count=genome_size)
  
  if genome.size != genome_size:
    raise ValueError(f"Expected genome of size {genome_size}, but got {genome.size}")
  
  return genome

def forward(genome: np.ndarray, state: np.ndarray, I: int, H: int, O: int):
  """
  Compute the forward pass of the policy network.
  """
  # Flatten the state just in case
  x = np.asarray(state, dtype=np.float32).reshape(I)

  # Extract weights and biases from genome
  p = 0
  W1 = genome[p : p + H * I].reshape(H, I); p += H * I
  b1 = genome[p : p + H].reshape(H,); p += H
  W2 = genome[p : p + H * H].reshape(H, H); p += H * H
  b2 = genome[p : p + H].reshape(H,); p += H
  W3 = genome[p : p + O * H].reshape(O, H); p += O * H
  b3 = genome[p : p + O].reshape(O,)

  # Forward pass
  h1 = np.tanh(W1 @ x + b1)
  h2 = np.tanh(W2 @ h1 + b2)
  y = W3 @ h2 + b3

  # Return index of the largest output (like C++ maxCoeff)
  return int(np.argmax(y))

def get_state():
  return np.array([0.5, 0.5, 0.5, 0, 0.01])

if __name__ == "__main__":
  state = get_state()
  genome = load_genome("genomes/0.net", I, H, O)
  action = forward(genome, state, I, H, O)
  print(f"Action: {action}")
