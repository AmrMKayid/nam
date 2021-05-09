import torch
import torch.nn as nn

from nam.models.base import Model
from nam.models.utils import init_weights


class DNN(Model):

    def __init__(
        self,
        config,
        name: str = "DNNModel",
        *,
        input_shape: int = 1,
        output_shape: int = 1,
    ) -> None:
        super(DNN, self).__init__(config, name)

        self.layers = []
        self.dropout = nn.Dropout(p=self.config.dropout)
        hidden_sizes = self.config.hidden_sizes

        self.layers.append(nn.Linear(input_shape, hidden_sizes[0], bias=True))
        self.layers.append(nn.ReLU())
        self.layers.append(self.dropout)

        for in_features, out_features in zip(hidden_sizes, hidden_sizes[1:]):
            self.layers.append(nn.Linear(in_features, out_features, bias=True))
            self.layers.append(nn.ReLU())
            self.layers.append(self.dropout)

        self.layers.append(nn.Linear(hidden_sizes[-1], output_shape, bias=True))

        self.model = nn.Sequential(*self.layers)
        self.apply(init_weights)

    def forward(self, inputs) -> torch.Tensor:
        return self.model(inputs), None
