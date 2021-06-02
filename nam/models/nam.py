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
    ) -> None:
        super(NAM, self).__init__(config, name)

        self._num_inputs = num_inputs
        self.dropout = nn.Dropout(p=self.config.dropout)

        if isinstance(num_units, list):
            assert len(num_units) == num_inputs
            self._num_units = num_units
        elif isinstance(num_units, int):
            self._num_units = [num_units for _ in range(self._num_inputs)]

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(config=config, name=f'FeatureNN_{i}', input_shape=1, num_units=self._num_units[i], feature_num=i)
            for i in range(num_inputs)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self._num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout(conc_out)

        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out
