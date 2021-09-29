import copy
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import nam.data
import nam.metrics
from nam.model import *
from nam.pl import *


class Config(SimpleNamespace):
    """Wrapper around SimpleNamespace, allows dot notation attribute access."""

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return Config(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else:
                setattr(self, key, val)


def defaults() -> Config:
    config = Config(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=2021,

        ## Data Path
        data_path="data/GALLUP.csv",
        experiment_name="NAM",
        regression=True,

        ## training
        num_epochs=10,
        learning_rate=0.00674,
        batch_size=1024,

        ## logs
        logdir="output",
        wandb=False,

        ## Hidden size for layers
        hidden_units=[64, 32, 32],

        ## Activation choice
        activation='exu',  ## Either `ExU` or `Relu`
        shallow_layer='exu',
        hidden_layer='relu',

        ## regularization_techniques
        dropout=0.5,
        feature_dropout=0.25,  #0.5,
        decay_rate=0.995,
        l2_regularization=1e-3,
        output_regularization=0.01,

        ## Num units for FeatureNN
        n_basis_functions=1024,
        units_multiplier=2,
        shuffle=True,

        ## Folded
        cross_val=False,
        n_folds=5,
        n_splits=3,
        id_fold=1,

        ## Models
        num_models=1,

        ## saver
        save_model_frequency=2,
        save_top_k=3,

        ## Early stopping
        use_dnn=False,
        early_stopping_patience=50,  ## For early stopping
    )

    return config


class NAMDataModule(pl.LightningDataModule):
    """Kitti Data Module It is specific to KITTI dataset i.e. dataloaders are
    for KITTI and Normalize transform uses the mean and standard deviation of
    this dataset."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train, (self.x_test, self.y_test) = nam.data.create_test_train_fold(dataset='sklearn_housing',
                                                                                 id_fold=config.id_fold,
                                                                                 n_folds=config.n_folds,
                                                                                 n_splits=config.n_splits,
                                                                                 regression=not config.regression)
        train_ds = list(self.train)
        x_trains, y_trains, x_validates, y_validates = [], [], [], []
        for fold in train_ds:
            (x_train, y_train), (x_validate, y_validate) = fold
            x_trains.append(torch.tensor(x_train))
            y_trains.append(torch.tensor(y_train))
            x_validates.append(torch.tensor(x_validate))
            y_validates.append(torch.tensor(y_validate))
        self.tmp_x_train = x_train

        x_train = torch.cat(x_trains)
        y_train = torch.cat(y_trains)
        x_validate = torch.cat(x_validates)
        y_validate = torch.cat(y_validates)

        self.train_ds = TensorDataset(x_train, y_train)
        self.val_ds = TensorDataset(x_validate, y_validate)
        self.test_ds = TensorDataset(torch.tensor(self.x_test), torch.tensor(self.y_test))

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size, shuffle=self.config.shuffle, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.config.batch_size,
                          shuffle=not self.config.shuffle,
                          num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.config.batch_size,
                          shuffle=not self.config.shuffle,
                          num_workers=8)


class NAM(pl.LightningModule):

    def __init__(self, config, x_train):
        super().__init__()
        self.config = config
        self.model = NeuralAdditiveModel(input_size=x_train.shape[-1],
                                         shallow_units=nam.data.calculate_n_units(x_train, config.n_basis_functions,
                                                                                  config.units_multiplier),
                                         hidden_units=list(map(int, config.hidden_units)),
                                         shallow_layer=ExULayer if config.shallow_layer == "exu" else ReLULayer,
                                         hidden_layer=ExULayer if config.hidden_layer == "exu" else ReLULayer,
                                         hidden_dropout=config.dropout,
                                         feature_dropout=config.feature_dropout)
        self.criterion = nam.metrics.penalized_mse if config.regression else nam.metrics.penalized_cross_entropy
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits, fnns_out = self.model(x)
        return logits, fnns_out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        # x = x.view(x.size(0), -1)
        logits, fnns_out = self(x)
        loss = self.criterion(logits, y, fnns_out, feature_penalty=self.config.output_regularization)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, fnns_out = self(x)
        loss = self.criterion(logits, y, fnns_out, feature_penalty=self.config.output_regularization)
        metric, score = nam.metrics.calculate_metric(logits, y, regression=self.config.regression)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            metric,
            score,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        return metric, score

    def test_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        x, y = batch
        logits, fnns_out = self(x)
        loss = self.criterion(logits, y, fnns_out, feature_penalty=self.config.output_regularization)
        metric, score = nam.metrics.calculate_metric(logits, y, regression=self.config.regression)
        self.log(
            'test_loss',
            loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f'test_{metric}',
            score,
            prog_bar=True,
            logger=True,
        )
        return loss, metric, score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.config.learning_rate,
                                      weight_decay=self.config.l2_regularization)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        return [optimizer], [scheduler]
