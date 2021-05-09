import torch

from nam.config.base import Config


def defaults() -> Config:
    config = Config(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=2021,

        ## Data Path
        data_path="data/GALLUP.csv",
        experiment_name="NAM",
        regression=False,

        ## training
        num_epochs=1,
        lr=3e-4,
        batch_size=1024,

        ## logs
        logdir="output",
        wandb=True,

        ## Hidden size for layers
        hidden_sizes=[64, 32],

        ## Activation choice
        activation='exu',  ## Either `ExU` or `Relu`
        optimizer='adam',

        ## regularization_techniques
        dropout=0.5,
        feature_dropout=0.5,
        decay_rate=0.995,
        l2_regularization=0.5,
        output_regularization=0.5,

        ## Num units for FeatureNN
        num_basis_functions=1000,
        units_multiplier=2,
        shuffle=True,

        ## Folded
        cross_val=False,
        num_folds=5,
        num_splits=3,
        fold_num=1,

        ## Models
        num_models=1,

        ## for dataloaders
        num_workers=16,

        ## saver
        save_model_frequency=2,
        save_top_k=3,

        ## Early stopping
        use_dnn=False,
        early_stopping_patience=50,  ## For early stopping
    )

    return config
