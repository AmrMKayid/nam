import torch
import torch.nn.functional as F


def ReLU(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
  """ReLU activation."""
  return F.relu(weights * (inputs - bias))
