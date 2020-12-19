import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ExU(torch.nn.Module):

  def __init__(
      self,
      in_features: int,
      out_features: int,  # equiv to num_units in tfnams
  ) -> None:
    super(ExU, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weights = Parameter(torch.Tensor(out_features, in_features))
    self.bias = Parameter(torch.Tensor(in_features))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    ## Page(4): initializing the weights using a normal distribution
    ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
    self.weights = torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
    self.bias = torch.nn.init.trunc_normal_(self.bias, std=0.5)

  def forward(
      self,
      inputs: torch.Tensor,
  ) -> torch.Tensor:
    # print(inputs, inputs.shape)
    output = (inputs - self.bias).matmul(torch.exp(self.weights).t())
    output = F.relu(output)  # ReLU activations capped at n (ReLU-n)

    return output

  def __repr__(self):
    return f'ExU(in_features={self.in_features}, out_features={self.out_features})'
