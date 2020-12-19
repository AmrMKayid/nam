import torch

from nam.config.base import Config


def defaults() -> Config:
  config = Config(
      device='cuda' if torch.cuda.is_available() else 'cpu',
      output_dir="output",
      training_epochs=10,
      lr=1e-2,
      batch_size=1024,

      ## regularization_techniques
      dropout=0.5,
      feature_dropout=0.0,
      decay_rate=0.995,
      l2_regularization=0.0,
      output_regularization=0.0,
      ## Num units for FeatureNN
      num_basis_functions=1000,
      units_multiplier=2,
      num_units=64,  ## TODO: number of hidden units? make it a list?!
      data_split=1,
      seed=1377,
      cross_val=False,
      n_models=1,
      num_splits=3,
      fold_num=1,
      activation='exu',
      shuffle=True,
      regression=False,
      debug=False,
      shallow=False,
      use_dnn=False,
      patience=10,  ## For early stopping
      n_folds=5,
      num_workers=16,  ## for dataloaders
  )

  return config
