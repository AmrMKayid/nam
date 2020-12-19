from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as sk_metrics


def accuracy(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
  """Accuracy for a binary classification model."""
  pred = model(inputs)
  binary_pred = pred > 0
  correct = binary_pred == (targets > 0.5)
  return correct.mean().float()


def rmse(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
  """Root mean squared error between true and predicted values."""
  pred = model(inputs)
  return float(np.sqrt(sk_metrics.mean_squared_error(targets, pred)))
