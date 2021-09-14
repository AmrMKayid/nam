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
        num_epochs=10,
        lr=1e-2,
        batch_size=128,

        ## logs
        logdir="output",
        wandb=False,

        ## Hidden size for layers
        hidden_sizes=[],  #[64, 32],

        ## Activation choice
        activation='exu',  ## Either `ExU` or `Relu`

        ## regularization_techniques
        dropout=0.1,
        feature_dropout=0.1,  #0.5,
        decay_rate=0.995,
        l2_regularization=0.1,
        output_regularization=0.1,

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
