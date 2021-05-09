from typing import List

import numpy as np
import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def get_num_units(
    config,
    features: torch.Tensor,
) -> List:
    features = features.cpu()
    num_unique_vals = [len(np.unique(features[:, i])) for i in range(features.shape[1])]

    num_units = [min(config.num_basis_functions, i * config.units_multiplier) for i in num_unique_vals]

    return num_units
