import argparse


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser("Neural Additive Models")

  parser.add_argument(
      '--training_epochs',
      default=10,
      type=str,
      help='The number of epochs to run training for.',
  )
  parser.add_argument(
      '--learning_rate',
      default=1e-2,
      type=float,
      help='Hyperparameter: learning rate.',
  )
  parser.add_argument(
      '--output_regularization',
      default=0.0,
      type=float,
      help='Hyperparameter: feature reg',
  )
  parser.add_argument(
      '--l2_regularization',
      default=0.0,
      type=float,
      help='Hyperparameter: l2 weight decay',
  )
  parser.add_argument(
      '--batch_size',
      default=1024,
      type=int,
      help='Hyperparameter: batch size.',
  )
  parser.add_argument(
      '--output_dir',
      default="output",
      type=str,
      help='Path to dir where to store summaries.',
  )
  parser.add_argument(
      '--dataset_name',
      default='Teleco',
      type=str,
      help='Name of the dataset to load for training.',
  )
  parser.add_argument(
      '--decay_rate',
      default=0.995,
      type=float,
      help='Hyperparameter: Optimizer decay rate',
  )
  parser.add_argument(
      '--dropout',
      default=0.5,
      type=float,
      help='Hyperparameter: Dropout rate',
  )
  parser.add_argument(
      '--data_split',
      default=1,
      type=int,
      help='Dataset split index to use. Possible '
      'values are 1 to `FLAGS.num_splits`.',
  )
  parser.add_argument(
      '--seed',
      default=1,
      type=int,
      help='seed for tf.',
  )
  parser.add_argument(
      '--feature_dropout',
      default=0.0,
      type=float,
      help='Hyperparameter: Prob. with which features are dropped',
  )
  parser.add_argument(
      '--num_basis_functions',
      default=1000,
      type=int,
      help='Number of basis functions '
      'to use in a FeatureNN for a real-valued feature.',
  )
  parser.add_argument(
      '--units_multiplier',
      default=2,
      type=int,
      help='Number of basis functions for a '
      'categorical feature',
  )
  parser.add_argument(
      '--cross_val',
      default=False,
      type=bool,
      help='Boolean flag indicating whether to '
      'perform cross validation or not.',
  )
  parser.add_argument(
      '--max_checkpoints_to_keep',
      default=1,
      type=int,
      help='Indicates the maximum '
      'number of recent checkpoint files to keep.',
  )
  parser.add_argument(
      '--save_checkpoint_every_n_epochs',
      default=10,
      type=int,
      help='Indicates the '
      'number of epochs after which an checkpoint is saved',
  )
  parser.add_argument(
      '--n_models',
      default=1,
      type=int,
      help='the number of models to train.',
  )
  parser.add_argument(
      '--num_splits',
      default=3,
      type=int,
      help='Number of data splits to use',
  )
  parser.add_argument(
      '--fold_num',
      default=1,
      type=int,
      help='Index of the fold to be used',
  )
  parser.add_argument(
      '--activation',
      default='exu',
      type=str,
      help='Activation function to used in the '
      'hidden layer. Possible options: (1) relu, (2) exu',
  )
  parser.add_argument(
      '--regression',
      default=False,
      type=bool,
      help='Boolean flag indicating whether we '
      'are solving a regression task or a classification task.',
  )
  parser.add_argument(
      '--debug',
      default=False,
      type=bool,
      help='Debug mode. Log additional things',
  )
  parser.add_argument(
      '--shallow',
      default=False,
      type=bool,
      help='Whether to use shallow or deep NN.',
  )
  parser.add_argument(
      '--use_dnn',
      default=False,
      type=bool,
      help='Deep NN baseline.',
  )
  parser.add_argument(
      '--early_stopping_epochs',
      default=60,
      type=int,
      help='Early stopping epochs',
  )
  parser.add_argument(
      '--n_folds',
      default=5,
      type=int,
      help='Number of N folds',
  )

  FLAGS = parser.parse_args([])
  return FLAGS
