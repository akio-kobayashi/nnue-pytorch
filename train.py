import argparse
import model as M
import nnue_dataset
import nnue_bin_dataset
import pytorch_lightning as pl
import features
import os
import torch
import typing
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader

class NNUEDataModule(pl.LightningDataModule):
    def __init__(self, train: str, val: str, features: str, num_workers: int = 1, batch_size: int = -1, smart_fen_skipping: bool = False, random_fen_skipping: int = 0, epoch_size: int = 10000000, py_data: bool = False):
        super().__init__()
        if batch_size <= 0:
            batch_size = 128 if not torch.cuda.is_available() else 8192
        self.save_hyperparameters()
        self.feature_set = features.get_feature_set_from_name(self.hparams.features)

    def setup(self, stage: typing.Optional[str] = None):
        if not os.path.exists(self.hparams.train):
            raise FileNotFoundError(f'{self.hparams.train} does not exist')
        if not os.path.exists(self.hparams.val):
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
            val_size = 1000000 # This was hardcoded in the original script
            self.val_ds = nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + self.hparams.batch_size - 1) // self.hparams.batch_size)

    def train_dataloader(self):
        if self.hparams.py_data:
            # In original script, num_workers was hardcoded to 4 for py_data
            return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)
        else:
            return DataLoader(self.train_ds, batch_size=None, batch_sampler=None)

    def val_dataloader(self):
        if self.hparams.py_data:
            # In original script, batch_size was hardcoded to 32 for py_data validation
            return DataLoader(self.val_ds, batch_size=32)
        else:
            return DataLoader(self.val_ds, batch_size=None, batch_sampler=None)

def main():
    # Process a few arguments manually before handing over to LightningCLI
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--threads", default=-1, type=int, help="Number of torch threads to use. Default automatic (cores).")
    args, _ = parser.parse_known_args()
    if args.threads > 0:
        print(f'limiting torch to {args.threads} threads.')
        t_set_num_threads(args.threads)

    # LightningCLI will add arguments for the model, datamodule, and trainer.
    # It will also handle seeding and checkpointing.
    # All model/data/trainer arguments are now passed through the command line
    # with dot notation, e.g., --model.lambda_ 0.5 or --data.batch_size 8192
    cli = LightningCLI(M.NNUE, NNUEDataModule, save_config_callback=None)

if __name__ == '__main__':
    main()