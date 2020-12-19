import pytorch_lightning as pl

from nam.config.default import defaults
from nam.data.base import NAMDataset
from nam.engine import Engine
from nam.models.nam import NAM
from nam.types import Config
from nam.utils.args import parse_args


def get_config() -> Config:
  args = parse_args()

  config = defaults()
  config.update(**vars(args))

  return config


def main():
  config = get_config()
  pl.seed_everything(config.seed)

  ## TODO: from static to args
  csv_file = 'data/GALLUP.csv'
  features_columns = ["income_2", "WP1219", "WP1220", "weo_gdpc_con_ppp"]
  targets_column = ["WP16"]
  weights_column = ["wgt"]
  dataset = NAMDataset(
      config=config,
      csv_file=csv_file,
      features_columns=features_columns,
      targets_column=targets_column,
      weights_column=weights_column,
      one_hot=False,
  )

  data_loaders = dataset.data_loaders(
      n_splits=config.num_splits,
      batch_size=config.batch_size,
      shuffle=config.shuffle,
      stratified=not config.regression,
      random_state=config.seed,
  )

  model = NAM(
      config=config,
      name="NAMModel",
      num_inputs=len(dataset[0][0]),
      num_units=config.num_units,
      shallow=config.shallow,
      feature_dropout=config.feature_dropout,
  ).to(device=config.device)

  train, val = next(iter(data_loaders))

  engine = Engine(config, model)

  trainer = pl.Trainer(default_root_dir=config.output_dir,)
  trainer.fit(engine, train)


if __name__ == "__main__":
  main()
