from typing import Sequence

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
      shallow: bool = True,
      feature_dropout: float = 0.0,
      dropout: float = 0.0,
      **kwargs,
  ) -> None:
    super().__init__(config, name)

    self._num_inputs = num_inputs
    if isinstance(num_units, list):
      assert len(num_units) == num_inputs
      self._num_units = num_units
    elif isinstance(num_units, int):
      self._num_units = [num_units for _ in range(self._num_inputs)]
    self._shallow = shallow
    self._feature_dropout = feature_dropout
    self._dropout = dropout
    self._kwargs = kwargs

    ## Builds the FeatureNNs on the first call.
    self.feature_nns = [None] * self._num_inputs
    for i in range(self._num_inputs):
      self.feature_nns[i] = FeatureNN(
          config=config,
          name=f'FeatureNN_{i}',
          num_units=self._num_units[i],
          input_shape=0,  #TODO:
          dropout=self._dropout,
          shallow=self._shallow,
          feature_num=i,
          **self._kwargs,
      )

    self._bias = torch.nn.Parameter(
        data=torch.zeros(1,),
        requires_grad=True,
    )

  def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
    """Returns the output computed by each feature net."""
    inputs_list = torch.split(inputs, self._num_inputs, dim=-1)
    return [
        self.feature_nns[i](input_i) for i, input_i in enumerate(inputs_list)
    ]

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    individual_outputs = self.calc_outputs(inputs)
    stacked_out = torch.stack(individual_outputs, dim=-1)
    dropout_out = F.dropout(stacked_out, p=self._feature_dropout)
    out = torch.sum(dropout_out, dim=-1)
    return out + self._bias
