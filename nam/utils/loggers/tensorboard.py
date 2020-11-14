"""Utilities for logging to the terminal."""

import os
import time

from torch.utils.tensorboard import SummaryWriter

from nam.utils.loggers import base


def _format_key(key: str) -> str:
  """Internal function for formatting keys in Tensorboard format."""
  return key.title().replace('_', '')


class TensorBoardLogger(base.Logger):
  """A simple `Pytorch`-friendly `TensorBoard` wrapper."""

  def __init__(
      self,
      log_dir: str,
      label: str = 'Logs',
  ) -> None:
    """Initializes the logger. Constructs a simple `TensorBoard` wrapper.

    Args:
      logdir: directory to which we should log files.
      label: label string to use when logging. Default to 'Logs'.
    """
    self._time = time.time()
    self.label = label
    self._iter = 0

    # Makes sure output directories exist.
    os.makedirs(log_dir, exist_ok=True)

    # Initialise the `TensorBoard` writers.
    self._summary_writter = SummaryWriter(log_dir=log_dir)

  def write(self, values: base.LoggingData):
    for key, value in values.items():
      self._summary_writter.scalar(
          f'{self.label}/{_format_key(key)}',
          value,
          step=self._iter,
      )
    self._iter += 1
