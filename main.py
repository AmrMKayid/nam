import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nam.config import defaults
from nam.data import FoldedDataset
from nam.data import NAMDataset
from nam.models import NAM
from nam.models import get_num_units
from nam.trainer import LitNAM
from nam.types import Config
from nam.utils import parse_args
from nam.utils import plot_mean_feature_importance
from nam.utils import plot_nams


def get_config() -> Config:
    args = parse_args()

    config = defaults()
    config.update(**vars(args))

    return config


def main():
    config = get_config()
    pl.seed_everything(config.seed)

    print(config)
    exit()

    if config.cross_val:
        dataset = FoldedDataset(
            config,
            data_path=config.data_path,
            features_columns=["income_2", "WP1219", "WP1220", "year", "weo_gdpc_con_ppp"],
            targets_column="WP16",
            weights_column="wgt",
        )
        dataloaders = dataset.train_dataloaders()

        model = NAM(
            config=config,
            name=config.experiment_name,
            num_inputs=len(dataset[0][0]),
            num_units=get_num_units(config, dataset.features),
        )

        for fold, (trainloader, valloader) in enumerate(dataloaders):

            # Folder hack
            tb_logger = TensorBoardLogger(save_dir=config.logdir, name=f'{model.name}', version=f'fold_{fold + 1}')

            checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}",
                                                  monitor='val_loss',
                                                  save_top_k=config.save_top_k,
                                                  mode='min')

            litmodel = LitNAM(config, model)
            trainer = pl.Trainer(logger=tb_logger,
                                 max_epochs=config.num_epochs,
                                 checkpoint_callback=checkpoint_callback)
            trainer.fit(litmodel, train_dataloader=trainloader, val_dataloaders=valloader)

            plot_mean_feature_importance(litmodel.model, dataset)
            plot_nams(litmodel.model, dataset, num_cols=1)
            plt.show()

    else:
        dataset = NAMDataset(
            config,
            data_path=config.data_path,
            features_columns=["income_2", "WP1219", "WP1220", "year", "weo_gdpc_con_ppp"],
            targets_column="WP16",
            weights_column="wgt",
        )
        trainloader, valloader, testloader = dataset.get_dataloaders()

        model = NAM(
            config=config,
            name=config.experiment_name,
            num_inputs=len(dataset[0][0]),
            num_units=get_num_units(config, dataset.features),
        )

        # Folder hack
        tb_logger = TensorBoardLogger(save_dir=config.logdir, name=f'{model.name}', version=f'0')

        checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}",
                                              monitor='val_loss',
                                              save_top_k=config.save_top_k,
                                              mode='min')

        litmodel = LitNAM(config, model)
        trainer = pl.Trainer(logger=tb_logger, max_epochs=config.num_epochs, checkpoint_callback=checkpoint_callback)
        trainer.fit(litmodel, train_dataloader=trainloader, val_dataloaders=valloader)

        plot_mean_feature_importance(litmodel.model, dataset)
        plot_nams(litmodel.model, dataset, num_cols=1)
        plt.show()


if __name__ == "__main__":
    main()
