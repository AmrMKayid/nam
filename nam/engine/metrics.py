import numpy as np
import torch
import torch.nn as nn


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
  """Accuracy for a binary classification model."""
  return (((targets.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() /
          targets.numel()).item()


def rmse(logits: torch.Tensor, targets: torch.Tensor) -> float:
  """Root mean squared error between true and predicted values."""
  return ((logits.view(-1) - targets.view(-1)).abs().sum() /
          logits.numel()).item()
