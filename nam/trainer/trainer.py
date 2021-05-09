from types import SimpleNamespace
from typing import Mapping
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm.autonotebook import tqdm

from nam.config import Config
from nam.models.saver import Checkpointer
from nam.trainer.losses import penalized_loss
from nam.trainer.metrics import accuracy
from nam.trainer.metrics import mae
from nam.utils.loggers import TensorBoardLogger


class Trainer:

    def __init__(self, config: SimpleNamespace, model: Sequence[nn.Module], dataset: torch.utils.data.Dataset) -> None:
        self.config = Config(**vars(config))  #config
        self.model = model
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.lr,
                                          weight_decay=self.config.decay_rate)

        self.writer = TensorBoardLogger(config)
        self.checkpointer = Checkpointer(model=model, config=config)

        self.criterion = lambda inputs, targets, weights, fnns_out, model: penalized_loss(
            self.config, inputs, targets, weights, fnns_out, model)

        self.metrics = lambda logits, targets: mae(logits, targets) if config.regression else accuracy(logits, targets)
        self.metrics_name = "MAE" if config.regression else "Accuracy"

        if config.wandb:
            wandb.watch(models=self.model, log='all', log_freq=10)

        self.dataloader_train, self.dataloader_val, \
          self.dataloader_test = self.dataset.get_dataloaders()

    def train_step(self, model: nn.Module, optimizer: optim.Optimizer, batch: torch.Tensor) -> torch.Tensor:
        """Performs a single gradient-descent optimization step."""

        features, targets = batch

        # Resets optimizer's gradients.
        self.optimizer.zero_grad()

        # Forward pass from the model.
        predictions, fnn_out = self.model(features)

        loss = self.criterion(predictions, targets, None, fnn_out, self.model)
        metrics = self.metrics(predictions, targets)

        # Backward pass.
        loss.backward()

        # Performs a gradient descent step.
        self.optimizer.step()

        return loss, metrics

    def train_epoch(self, model: nn.Module, optimizer: optim.Optimizer,
                    dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Performs an epoch of gradient descent optimization on
        `dataloader`."""
        model.train()
        loss = 0.0
        metrics = 0.0
        with tqdm(dataloader, leave=False) as pbar:
            for batch in pbar:

                # Performs a gradient-descent step.
                step_loss, step_metrics = self.train_step(model, optimizer, batch)
                loss += step_loss
                metrics += step_metrics

                pbar.set_description(f"TL Step: {step_loss:.3f} | {self.metrics_name}: {step_metrics:.3f}")

        return loss / len(dataloader), metrics / len(dataloader)

    def evaluate_step(self, model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluates `model` on a `batch`."""

        features, targets = batch

        # Forward pass from the model.
        predictions, fnn_out = self.model(features)

        # Calculates loss on mini-batch.
        loss = self.criterion(predictions, targets, None, fnn_out, self.model)
        metrics = self.metrics(predictions, targets)

        # self.writer.write({"val_loss_step": loss.detach().cpu().numpy().item()})

        return loss, metrics

    def evaluate_epoch(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Performs an evaluation of the `model` on the `dataloader."""
        model.eval()
        loss = 0.0
        metrics = 0.0
        with tqdm(dataloader, leave=False) as pbar:
            for batch in pbar:
                # Accumulates loss in dataset.
                with torch.no_grad():
                    # step_loss = self.evaluate_step(model, batch, pbar)
                    # loss += self.evaluate_step(model, batch, pbar)
                    step_loss, step_metrics = self.evaluate_step(model, batch)
                    loss += step_loss
                    metrics += step_metrics

                    pbar.set_description((f"VL Step: {step_loss:.3f} | {self.metrics_name}: {step_metrics:.3f}"))

        return loss / len(dataloader), metrics / len(dataloader)

    def train(self):
        num_epochs = self.config.num_epochs

        with tqdm(range(num_epochs)) as pbar_epoch:
            for epoch in pbar_epoch:
                # Trains model on whole training dataset, and writes on `TensorBoard`.
                loss_train, metrics_train = self.train_epoch(self.model, self.optimizer, self.dataloader_train)
                self.writer.write({
                    "loss_train_epoch": loss_train.detach().cpu().numpy().item(),
                    f"{self.metrics_name}_train_epoch": metrics_train,
                })

                # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
                loss_val, metrics_val = self.evaluate_epoch(self.model, self.dataloader_val)
                self.writer.write({
                    "loss_val_epoch": loss_val.detach().cpu().numpy().item(),
                    f"{self.metrics_name}_val_epoch": metrics_val,
                })

                # Checkpoints model weights.
                if epoch % self.config.save_model_frequency == 0:
                    self.checkpointer.save(epoch)

                # Updates progress bar description.
                pbar_epoch.set_description(f"""Epoch({epoch}):
            TL: {loss_train.detach().cpu().numpy().item():.3f} |
            VL: {loss_val.detach().cpu().numpy().item():.3f} |
            {self.metrics_name}: {metrics_train:.3f}""")

    def test(self):
        num_epochs = self.config.num_epochs

        with tqdm(range(num_epochs)) as pbar_epoch:
            for epoch in pbar_epoch:

                # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
                loss_test, metrics_test = self.evaluate_epoch(self.model, self.dataloader_test)
                # tune.report(loss_test=loss_test.detach().cpu().numpy().item())
                self.writer.write({
                    "loss_test_epoch": loss_test.detach().cpu().numpy().item(),
                    f"{self.metrics_name}_test_epoch": metrics_test,
                })

                # Updates progress bar description.
                pbar_epoch.set_description("Test Loss: {:.2f} ".format(loss_test.detach().cpu().numpy().item()))
