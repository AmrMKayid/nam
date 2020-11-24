import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ExU(torch.nn.Module):

  def __init__(
      self,
      in_dim: int,
      out_dim: int,  # equiv to num_units in tfnams
  ) -> None:
    super(ExU, self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.weights = Parameter(torch.Tensor(out_dim, in_dim))
    self.bias = Parameter(torch.Tensor(in_dim))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    self.weights = torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
    self.bias = torch.nn.init.trunc_normal_(self.bias, std=0.5)

  def forward(
      self,
      inputs: torch.Tensor,
  ) -> torch.Tensor:
    output = (inputs - self.bias).matmul(torch.exp(self.weights).t())
    output = F.relu(output)

    return output
