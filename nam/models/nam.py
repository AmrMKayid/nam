from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nam.models.base import Model
from nam.models.featurenn import FeatureNN


class NAM(Model):

  def __init__(
      self,
      config,
      name,
      *,
      num_inputs: int,
      num_units: int,
      num_outputs=None,
  ) -> None:
    super(NAM, self).__init__(config, name)

    self._num_inputs = num_inputs
    self._num_outputs = num_outputs
    if isinstance(num_units, list):
      assert len(num_units) == num_inputs
      self._num_units = num_units
    elif isinstance(num_units, int):
      self._num_units = [num_units for _ in range(self._num_inputs)]

    ## Builds the FeatureNNs on the first call.
    self.feature_nns = nn.Sequential()
    for i in range(self._num_inputs):
      self.feature_nns.add_module(
          f'FeatureNN_{i}',
          FeatureNN(
              config=config,
              name=f'FeatureNN_{i}',
              input_shape=1,
              num_units=self._num_units[i],
              feature_num=i,
          ))

    self._bias = torch.nn.Parameter(data=torch.zeros(1,), requires_grad=True)

    if num_outputs:
      self.linear = nn.Linear(in_features=len(num_units),
                              out_features=num_outputs)

  def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
    """Returns the output computed by each feature net."""
    fnns = dict(self.feature_nns.named_children())

    inputs_tuple = torch.chunk(inputs, self._num_inputs, dim=-1)
    return [
        fnns[f"FeatureNN_{index}"](input_i)
        for index, input_i in enumerate(inputs_tuple)
    ]

  def forward(
      self,
      inputs: torch.Tensor,
      weights: torch.Tensor = torch.tensor(1),
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    individual_outputs = self.calc_outputs(inputs)
    stacked_out = torch.stack(individual_outputs, dim=-1).squeeze()
    dropout_out = F.dropout(stacked_out, p=self.config.feature_dropout)
    if self._num_outputs:
      logits = self.linear(dropout_out) + self._bias
      # logits *= torch.repeat_interleave(weights,
      #                                   logits.shape[-1]).view(logits.shape)
      preds = F.softmax(logits, dim=-1)
      return preds, dropout_out

    out = torch.sum(dropout_out, dim=-1)
    return (out + self._bias).squeeze(), dropout_out
