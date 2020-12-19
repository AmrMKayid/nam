# NAM: Neural Additive Models - Interpretable Machine Learning with Neural Nets

  **[Overview](#overview)**
| **[Installation](#installation)**
| **[Paper](https://arxiv.org/pdf/2004.13912.pdf)**

![PyPI Python Version](https://img.shields.io/pypi/pyversions/nam)
![PyPI version](https://badge.fury.io/py/nam.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2004.13912-b31b1b.svg)](https://arxiv.org/abs/2004.13912)
[![GitHub license](https://img.shields.io/pypi/l/nam)](./LICENSE)

NAM is a library for generalized additive models research.
Neural Additive Models (NAMs) combine some of the expressivity of DNNs with the inherent intelligibility of generalized additive models. NAMs learn a linear combination of neural networks that each attend to a single input feature. These networks are trained jointly and can learn arbitrarily complex relationships between their input feature and the output.

## Overview

> TODO:

## Usage

```bash
$ python main.py -h
usage: Neural Additive Models [-h] [--training_epochs TRAINING_EPOCHS]
                              [--learning_rate LEARNING_RATE]
                              [--output_regularization OUTPUT_REGULARIZATION]
                              [--l2_regularization L2_REGULARIZATION]
                              [--batch_size BATCH_SIZE] [--logdir LOGDIR]
                              [--dataset_name DATASET_NAME]
                              [--decay_rate DECAY_RATE] [--dropout DROPOUT]
                              [--data_split DATA_SPLIT] [--seed SEED]
                              [--feature_dropout FEATURE_DROPOUT]
                              [--num_basis_functions NUM_BASIS_FUNCTIONS]
                              [--units_multiplier UNITS_MULTIPLIER]
                              [--cross_val CROSS_VAL]
                              [--max_checkpoints_to_keep MAX_CHECKPOINTS_TO_KEEP]
                              [--save_checkpoint_every_n_epochs SAVE_CHECKPOINT_EVERY_N_EPOCHS]
                              [--n_models N_MODELS] [--num_splits NUM_SPLITS]
                              [--fold_num FOLD_NUM] [--activation ACTIVATION]
                              [--regression REGRESSION] [--debug DEBUG]
                              [--shallow SHALLOW] [--use_dnn USE_DNN]
                              [--early_stopping_epochs EARLY_STOPPING_EPOCHS]
                              [--n_folds N_FOLDS]

optional arguments:
  -h, --help            show this help message and exit
  --training_epochs TRAINING_EPOCHS
                        The number of epochs to run training for.
  --learning_rate LEARNING_RATE
                        Hyperparameter: learning rate.
  --output_regularization OUTPUT_REGULARIZATION
                        Hyperparameter: feature reg
  --l2_regularization L2_REGULARIZATION
                        Hyperparameter: l2 weight decay
  --batch_size BATCH_SIZE
                        Hyperparameter: batch size.
  --logdir LOGDIR       Path to dir where to store summaries.
  --dataset_name DATASET_NAME
                        Name of the dataset to load for training.
  --decay_rate DECAY_RATE
                        Hyperparameter: Optimizer decay rate
  --dropout DROPOUT     Hyperparameter: Dropout rate
  --data_split DATA_SPLIT
                        Dataset split index to use. Possible values are 1 to
                        `FLAGS.num_splits`.
  --seed SEED           seed for tf.
  --feature_dropout FEATURE_DROPOUT
                        Hyperparameter: Prob. with which features are dropped
  --num_basis_functions NUM_BASIS_FUNCTIONS
                        Number of basis functions to use in a FeatureNN for a
                        real-valued feature.
  --units_multiplier UNITS_MULTIPLIER
                        Number of basis functions for a categorical feature
  --cross_val CROSS_VAL
                        Boolean flag indicating whether to perform cross
                        validation or not.
  --max_checkpoints_to_keep MAX_CHECKPOINTS_TO_KEEP
                        Indicates the maximum number of recent checkpoint
                        files to keep.
  --save_checkpoint_every_n_epochs SAVE_CHECKPOINT_EVERY_N_EPOCHS
                        Indicates the number of epochs after which an
                        checkpoint is saved
  --n_models N_MODELS   the number of models to train.
  --num_splits NUM_SPLITS
                        Number of data splits to use
  --fold_num FOLD_NUM   Index of the fold to be used
  --activation ACTIVATION
                        Activation function to used in the hidden layer.
                        Possible options: (1) relu, (2) exu
  --regression REGRESSION
                        Boolean flag indicating whether we are solving a
                        regression task or a classification task.
  --debug DEBUG         Debug mode. Log additional things
  --shallow SHALLOW     Whether to use shallow or deep NN.
  --use_dnn USE_DNN     Deep NN baseline.
  --early_stopping_epochs EARLY_STOPPING_EPOCHS
                        Early stopping epochs
  --n_folds N_FOLDS     Number of N folds
```


## Citing NAM


```bibtex
@misc{kayid2020nams,
  title={Neural additive models Library},
  author={Kayid, Amr and Frosst, Nicholas and Hinton, Geoffrey E},
  year={2020}
}
```

```bibtex
@article{agarwal2020neural,
  title={Neural additive models: Interpretable machine learning with neural nets},
  author={Agarwal, Rishabh and Frosst, Nicholas and Zhang, Xuezhou and Caruana, Rich and Hinton, Geoffrey E},
  journal={arXiv preprint arXiv:2004.13912},
  year={2020}
}
```
