import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LinReLU(torch.nn.Module):
  __constants__ = ['bias']

  def __init__(
      self,
      in_features: int,
      out_features: int,
  ) -> None:
    super(LinReLU, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(torch.Tensor(out_features, in_features),
                            requires_grad=True)
    self.bias = Parameter(torch.Tensor(out_features), requires_grad=True)

    self.reset_parameters()

  def reset_parameters(self) -> None:
    nn.init.kaiming_normal_(self.weight)
    nn.init.constant_(self.bias, 0.1)

  def forward(
      self,
      inputs: torch.Tensor,
  ) -> torch.Tensor:
    output = F.linear(inputs, self.weight, self.bias)
    output = F.relu(output)

    return output

  def extra_repr(self):
    return f'in_features={self.in_features}, out_features={self.out_features}'
