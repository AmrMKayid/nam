import torch

from nam.config.base import Config


def defaults() -> Config:
  config = Config(
      device='cuda' if torch.cuda.is_available() else 'cpu',
      logdir="logs",
      lr=1e-2,
      batch_size=1024,
      l2_regularization=0.0,
      output_regularization=0.0,
      decay_rate=0.995,
      dropout=0.5,
      feature_dropout=0.0,
      data_split=1,
      seed=1377,
      num_basis_functions=1000,
      units_multiplier=2,
      cross_val=False,
      max_checkpoints_to_keep=1,  # TODO: checkpointer
      save_checkpoint_every_n_epochs=10,
      n_models=1,
      num_splits=3,
      fold_num=1,
      activation='exu',
      regression=False,
      debug=False,
      shallow=False,
      use_dnn=False,
      early_stopping_epochs=60,
      n_folds=5,
  )

  return config
