from pathlib import Path
from typing import Optional

import dlshogi_dataset
import dlshogi_transformer as M
import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningCLI
from torch import set_num_threads as t_set_num_threads
from torch.utils.data import DataLoader


DEFAULT_EPOCH_SIZE = 2_000_000
DEFAULT_VALIDATION_SIZE = 200_000


def _default_batch_size() -> int:
    return 64 if not torch.cuda.is_available() else 256


torch.set_float32_matmul_precision("high")


class DLShogiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: str,
        val: str,
        num_workers: int = 4,
        val_num_workers: int = 1,
        batch_size: int = -1,
        smart_fen_skipping: bool = False,
        random_fen_skipping: int = 0,
        epoch_size: int = DEFAULT_EPOCH_SIZE,
        validation_size: int = DEFAULT_VALIDATION_SIZE,
        threads: int = -1,
    ) -> None:
        super().__init__()
        if threads > 0:
            print(f"limiting torch to {threads} threads.")
            t_set_num_threads(threads)
        if batch_size <= 0:
            batch_size = _default_batch_size()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if not Path(self.hparams.train).exists():
            raise FileNotFoundError(f"{self.hparams.train} does not exist")
        if not Path(self.hparams.val).exists():
            raise FileNotFoundError(f"{self.hparams.val} does not exist")

        train_infinite = dlshogi_dataset.DlshogiBatchDataset(
            self.hparams.train,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            filtered=self.hparams.smart_fen_skipping,
            random_fen_skipping=self.hparams.random_fen_skipping,
        )
        val_infinite = dlshogi_dataset.DlshogiBatchDataset(
            self.hparams.val,
            self.hparams.batch_size,
            cyclic=False,
            num_workers=self.hparams.val_num_workers,
            filtered=False,
            random_fen_skipping=0,
        )
        self.train_ds = dlshogi_dataset.FixedNumBatchesDataset(
            train_infinite,
            (self.hparams.epoch_size + self.hparams.batch_size - 1)
            // self.hparams.batch_size,
        )
        self.val_ds = dlshogi_dataset.FixedNumBatchesDataset(
            val_infinite,
            (self.hparams.validation_size + self.hparams.batch_size - 1)
            // self.hparams.batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=None, batch_sampler=None)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=None, batch_sampler=None)


def main():
    cli_kwargs = {"save_config_callback": None}
    try:
        LightningCLI(
            M.DLShogiTransformer,
            DLShogiDataModule,
            parser_kwargs={"fit": {"default_config_files": ["config_dlshogi_transformer.yaml"]}},
            **cli_kwargs,
        )
    except TypeError as exc:
        if "default_config_files" not in str(exc):
            raise
        LightningCLI(M.DLShogiTransformer, DLShogiDataModule, **cli_kwargs)


if __name__ == "__main__":
    main()
