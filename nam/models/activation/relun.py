import torch


def ReLUN(
    inputs: torch.Tensor,
    n: int = 1,
) -> torch.Tensor:
  """ReLU activation clipped at n."""
  return torch.clamp(inputs, 0, n)
