import torch


def ExU(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
  """ExU hidden unit modification."""
  return torch.exp(weights) * (inputs - bias)
