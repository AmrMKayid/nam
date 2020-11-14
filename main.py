import time
import json
import os
import random
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from loguru import logger

from nam.config.default import defaults
from nam.types import Config
from nam.utils.args import parse_args


def get_config() -> Config:

  args = parse_args()

  config = defaults()
  config.update(**vars(args))

  # Creates the necessary output directory.
  config.output_dir = os.path.abspath(config.output_dir)
  os.makedirs(config.output_dir, exist_ok=True)
  logger.debug(f'Creating output dir `{config.output_dir}`')

  run_exp_folder = os.path.join(config.output_dir,
                                "Exp_" + time.strftime('%Y-%m-%d_%H%M%S'))
  os.makedirs(run_exp_folder, exist_ok=True)
  logger.debug(f'Creating run exp folder `{run_exp_folder}`')

  config.log_dir = os.path.join(run_exp_folder, "logs")
  os.makedirs(config.log_dir, exist_ok=True)

  config.ckpt_dir = os.path.join(run_exp_folder, "ckpts")
  os.makedirs(config.ckpt_dir, exist_ok=True)
  config.model_path = os.path.abspath(os.path.join(config.ckpt_dir, 'model.th'))

  # Save the configuration in a config.json file
  with open(os.path.join('.', 'config.json'), 'w') as f:
    json.dump(vars(config), f, indent=2, default=lambda o: o.__dict__)
  logger.info('Saving configuration file in `{0}`'.format(
      os.path.abspath(os.path.join('.', 'config.json'))))

  config.device = torch.device(config.device)

  return config


def init_random_seeds(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    logger.debug("PyTorch is using Cuda...")
    torch.cuda.manual_seed_all(seed)
  else:
    logger.debug("PyTorch is using CPU...")


def main():
  config = get_config()

  init_random_seeds(config.seed)


if __name__ == "__main__":
  main()
