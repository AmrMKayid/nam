"""Utilities for logging to the terminal."""

import os
import time

import wandb
from torch.utils.tensorboard import SummaryWriter

from nam.utils.loggers import base


def _format_key(key: str) -> str:
    """Internal function for formatting keys in Tensorboard format."""
    return key.title().replace('_', '')


class TensorBoardLogger(base.Logger):
    """A simple `Pytorch`-friendly `TensorBoard` wrapper."""

    def __init__(self, config: dict = None, project: str = "nam", label: str = 'Logs') -> None:
        """Initializes the logger. Constructs a simple `TensorBoard` wrapper.

        Args:
          logdir: directory to which we should log files.
          label: label string to use when logging. Default to 'Logs'.
        """
        self._time = time.time()
        self.label = label
        self.config = config
        self._iter = 0

        # Makes sure output directories exist.
        log_dir = os.path.join(config.logdir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Initialise the `TensorBoard` writers.
        self._summary_writter = SummaryWriter(log_dir=log_dir)

        if config.wandb:
            wandb.init(project=project, config=vars(config), reinit=True, magic=True)

    def write(self, values: base.LoggingData):
        for key, value in values.items():
            self._summary_writter.add_scalar(f'{self.label}/{_format_key(key)}', value, global_step=self._iter)
            if self.config.wandb:
                wandb.log({f'{self.label}/{_format_key(key)}': value})

        self._iter += 1
