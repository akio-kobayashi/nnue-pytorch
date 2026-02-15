import model as M
import nnue_dataset
import nnue_bin_dataset
import pytorch_lightning as pl
import features as features_module
import torch
import typing
from pathlib import Path
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader

class NNUEDataModule(pl.LightningDataModule):
    def __init__(self, train: str, val: str, features: str, num_workers: int = 1, batch_size: int = -1, smart_fen_skipping: bool = False, random_fen_skipping: int = 0, epoch_size: int = 10000000, validation_size: int = 1000000, py_data: bool = False, py_data_train_num_workers: int = 4, py_data_val_batch_size: int = 32, threads: int = -1):
        super().__init__()
        if threads > 0:
            print(f'limiting torch to {threads} threads.')
            t_set_num_threads(threads)
        if batch_size <= 0:
            batch_size = 128 if not torch.cuda.is_available() else 8192
        self.save_hyperparameters()
        self.feature_set = features_module.get_feature_set_from_name(self.hparams.features)

    def setup(self, stage: typing.Optional[str] = None):
        if not Path(self.hparams.train).exists():
            raise FileNotFoundError(f'{self.hparams.train} does not exist')
        if not Path(self.hparams.val).exists():
            raise FileNotFoundError(f'{self.hparams.val} does not exist')

        # The C++ data loader needs a device.
        main_device = 'cpu'
        if self.trainer:
            if self.trainer.strategy.root_device.type == 'cuda':
                main_device = f'cuda:{self.trainer.strategy.root_device.index}'

        if self.hparams.py_data:
            self.train_ds = nnue_bin_dataset.NNUEBinData(self.hparams.train, self.feature_set)
            self.val_ds = nnue_bin_dataset.NNUEBinData(self.hparams.val, self.feature_set)
        else:
            train_infinite = nnue_dataset.SparseBatchDataset(self.hparams.features, self.hparams.train, self.hparams.batch_size, num_workers=self.hparams.num_workers,
                                                           filtered=self.hparams.smart_fen_skipping, random_fen_skipping=self.hparams.random_fen_skipping, device=main_device)
            val_infinite = nnue_dataset.SparseBatchDataset(self.hparams.features, self.hparams.val, self.hparams.batch_size, filtered=self.hparams.smart_fen_skipping,
                                                           random_fen_skipping=self.hparams.random_fen_skipping, device=main_device)
            self.train_ds = nnue_dataset.FixedNumBatchesDataset(train_infinite, (self.hparams.epoch_size + self.hparams.batch_size - 1) // self.hparams.batch_size)
            val_size = self.hparams.validation_size
            self.val_ds = nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + self.hparams.batch_size - 1) // self.hparams.batch_size)

    def train_dataloader(self):
        if self.hparams.py_data:
            return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.py_data_train_num_workers)
        else:
            return DataLoader(self.train_ds, batch_size=None, batch_sampler=None)

    def val_dataloader(self):
        if self.hparams.py_data:
            return DataLoader(self.val_ds, batch_size=self.hparams.py_data_val_batch_size)
        else:
            return DataLoader(self.val_ds, batch_size=None, batch_sampler=None)

class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--gamma", type=float, default=0.992, help="各エポック後に学習率に乗じる乗法的係数")
        parser.add_argument("--num_buckets", type=int, default=8, help="LayerStackのバケット数")
        parser.link_arguments("gamma", "model.gamma")
        parser.link_arguments("num_buckets", "model.num_buckets")
        parser.link_arguments("data.features", "model.features")


def main():
    # LightningCLI will add arguments for the model, datamodule, and trainer.
    # It will also handle seeding and checkpointing.
    # All model/data/trainer arguments are now passed through the command line
    # with dot notation, e.g., --model.lambda_ 0.5 or --data.batch_size 8192
    cli = MyCLI(M.NNUE, NNUEDataModule, save_config_callback=None, default_config_files=['config.yaml'])

if __name__ == '__main__':
    main()
