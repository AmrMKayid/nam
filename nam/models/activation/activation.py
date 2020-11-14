import torch
import torch.nn as nn

from .exu import ExU
from .relu import ReLU
from .relun import ReLUN


def truncated_normal_(
    tensor: torch.Tensor,
    mean: float = 0,
    std: float = 1,
) -> torch.Tensor:
  """source: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/19"""
  size = tensor.shape
  tmp = tensor.new_empty(size + (4,)).normal_()
  valid = (tmp < 2) & (tmp > -2)
  ind = valid.max(-1, keepdim=True)[1]
  tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
  tensor.data.mul_(std).add_(mean)
  return tensor


class ActivationLayer(nn.Module):
  """Custom activation Layer to support ExU hidden units."""

  def __init__(
      self,
      *,
      num_units: int,
      input_shape: int,
      activation: str = 'exu',
      name: str = 'ActivationLayer',
  ) -> None:
    super(ActivationLayer, self).__init__()

    self.num_units = num_units
    self.input_shape = input_shape
    self.name = name

    if activation == 'relu':
      self._activation = lambda inputs, weight, bias: ReLU(
          inputs, weight, bias)  #TODO: check this with nick
      self._beta_initializer = lambda tensor: nn.init.xavier_uniform_(tensor)
    elif activation == 'exu':
      self._activation = lambda inputs, weight, bias: ReLUN(
          ExU(inputs, weight, bias))
      self._beta_initializer = lambda tensor: truncated_normal_(
          tensor,
          mean=4.0,
          std=0.5,
      )
    else:
      raise ValueError('{} is not a valid activation'.format(activation))

    self._beta = torch.nn.Parameter(
        data=self._beta_initializer(
            torch.zeros(self.input_shape[-1], self.num_units)),
        requires_grad=True,
    )
    self._c = torch.nn.Parameter(
        data=truncated_normal_(
            torch.zeros(1, self.num_units),
            std=0.5,
        ),
        requires_grad=True,
    )

  def forward(self, inputs) -> torch.Tensor:
    center = self._c.repeat(inputs.shape[0], 1)
    out = self._activation(inputs, self._beta, center)
    return out
