"""Utilities for logging to Weights & Biases."""

import torch
import wandb

from nam.utils.loggers import base


class WandBLogger(base.Logger):
    """Logs to a `wandb` dashboard."""

    def __init__(
        self,
        project: str = "nam",
        configs: dict = None,
    ) -> None:
        super().__init__(log_dir=configs.logdir)
        wandb.init(project=project, config=configs)

    def write(self, data: base.LoggingData) -> None:
        wandb.log(data)

    def watch(
        self,
        model: torch.nn.Module,
        **kwargs: dict,
    ) -> None:
        wandb.watch(model, kwargs)
