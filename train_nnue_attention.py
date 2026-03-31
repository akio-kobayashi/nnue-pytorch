import nnue_attention as M
import nnue_dataset
import pytorch_lightning as pl
import features as features_module
import torch
from pathlib import Path
from typing import Optional
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader


DEFAULT_EPOCH_SIZE = 10_000_000
DEFAULT_VALIDATION_SIZE = 1_000_000


def _default_batch_size() -> int:
    return 128 if not torch.cuda.is_available() else 8192


if hasattr(torch.backends, "cuda"):
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


class ResettableFixedNumBatchesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_batches):
        super().__init__()
        self.dataset = dataset
        self.iter = None
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx == 0 or self.iter is None:
            self.iter = iter(self.dataset)
        return next(self.iter)


class NNUEAttentionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: str,
        val: str,
        features: str,
        num_workers: int = 1,
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
        self.feature_set = features_module.get_feature_set_from_name(self.hparams.features)

    def setup(self, stage: Optional[str] = None) -> None:
        if not Path(self.hparams.train).exists():
            raise FileNotFoundError(f"{self.hparams.train} does not exist")
        if not Path(self.hparams.val).exists():
            raise FileNotFoundError(f"{self.hparams.val} does not exist")

        main_device = "cpu"
        if self.trainer and self.trainer.strategy.root_device.type == "cuda":
            main_device = f"cuda:{self.trainer.strategy.root_device.index}"

        train_infinite = nnue_dataset.SparseBatchDataset(
            self.hparams.features,
            self.hparams.train,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            filtered=self.hparams.smart_fen_skipping,
            random_fen_skipping=self.hparams.random_fen_skipping,
            device=main_device,
        )
        val_infinite = nnue_dataset.SparseBatchDataset(
            self.hparams.features,
            self.hparams.val,
            self.hparams.batch_size,
            cyclic=False,
            num_workers=self.hparams.val_num_workers,
            filtered=False,
            random_fen_skipping=0,
            device=main_device,
        )

        self.train_ds = ResettableFixedNumBatchesDataset(
            train_infinite,
            (self.hparams.epoch_size + self.hparams.batch_size - 1) // self.hparams.batch_size,
        )
        self.val_ds = ResettableFixedNumBatchesDataset(
            val_infinite,
            (self.hparams.validation_size + self.hparams.batch_size - 1) // self.hparams.batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=None, batch_sampler=None)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=None, batch_sampler=None)


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("data.features", "model.features")


def main():
    cli_kwargs = {"save_config_callback": None}
    try:
        MyCLI(
            M.NNUEAttention,
            NNUEAttentionDataModule,
            parser_kwargs={"fit": {"default_config_files": ["config_nnue_attention.yaml"]}},
            **cli_kwargs,
        )
    except TypeError as exc:
        if "default_config_files" not in str(exc):
            raise
        MyCLI(M.NNUEAttention, NNUEAttentionDataModule, **cli_kwargs)


if __name__ == "__main__":
    main()
