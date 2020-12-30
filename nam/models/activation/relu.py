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
    self.weight = Parameter(torch.Tensor(out_features, in_features))
    self.bias = Parameter(torch.Tensor(out_features))

    self.reset_parameters()

  def reset_parameters(self) -> None:
    nn.init.kaiming_normal_(self.weight)
    if self.bias is not None:
      nn.init.zeros_(self.bias)

  def forward(
      self,
      inputs: torch.Tensor,
  ) -> torch.Tensor:
    output = F.linear(inputs, self.weight, self.bias)
    output = F.relu(output)

    return output

  def extra_repr(self):
    return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None)
