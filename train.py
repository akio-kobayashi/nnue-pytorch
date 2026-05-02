import model as M
import nnue_dataset
import nnue_bin_dataset
import preference_dataset
import pickle
import os
import pytorch_lightning as pl
import features as features_module
import torch
import sys
import yaml
from pathlib import Path
from typing import Any, Optional
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader


DEFAULT_EPOCH_SIZE = 10_000_000
DEFAULT_VALIDATION_SIZE = 1_000_000
DEFAULT_PY_DATA_TRAIN_NUM_WORKERS = 4
DEFAULT_PY_DATA_VAL_BATCH_SIZE = 32


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _to_serializable(value.as_dict())
    if hasattr(value, "__dict__"):
        return _to_serializable(vars(value))
    return str(value)


def _flatten_for_logger(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_for_logger(child, child_prefix))
        return flat
    if isinstance(value, list):
        return {prefix: ",".join(str(_to_serializable(v)) for v in value)}
    return {prefix: value}


class HParamsSnapshotCallback(pl.Callback):
    def __init__(self, cli_config: Any) -> None:
        super().__init__()
        self.cli_config = _to_serializable(cli_config)
        self._written = False

    def _log_dir(self, trainer: pl.Trainer) -> Path:
        logger = trainer.logger
        if logger is not None and getattr(logger, "log_dir", None):
            return Path(logger.log_dir)
        return Path(trainer.default_root_dir)

    def _trainer_summary(self, trainer: pl.Trainer) -> dict[str, Any]:
        return {
            "accelerator": trainer.accelerator.__class__.__name__.replace("Accelerator", "").lower(),
            "devices": trainer.num_devices,
            "max_epochs": trainer.max_epochs,
            "precision": str(trainer.precision),
            "default_root_dir": trainer.default_root_dir,
            "accumulate_grad_batches": trainer.accumulate_grad_batches,
        }

    def _merged_config(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> dict[str, Any]:
        config = self.cli_config if isinstance(self.cli_config, dict) else {"config": self.cli_config}
        merged = dict(config)
        merged["model_hparams"] = _to_serializable(dict(pl_module.hparams))
        datamodule = trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "hparams"):
            merged["data_hparams"] = _to_serializable(dict(datamodule.hparams))
        merged["trainer_runtime"] = self._trainer_summary(trainer)
        return merged

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self._written or not trainer.is_global_zero or stage != "fit":
            return
        config = self._merged_config(trainer, pl_module)
        log_dir = self._log_dir(trainer)
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "hparams.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(_flatten_for_logger(config))
        self._written = True


def _default_batch_size() -> int:
    return 128 if not torch.cuda.is_available() else 8192


class NNUEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: str,
        val: str,
        features: str,
        num_workers: int = 1,
        batch_size: int = -1,
        smart_fen_skipping: bool = False,
        random_fen_skipping: int = 0,
        epoch_size: int = DEFAULT_EPOCH_SIZE,
        validation_size: int = DEFAULT_VALIDATION_SIZE,
        py_data: bool = False,
        py_data_train_num_workers: int = DEFAULT_PY_DATA_TRAIN_NUM_WORKERS,
        py_data_val_batch_size: int = DEFAULT_PY_DATA_VAL_BATCH_SIZE,
        py_data_sampling_mode: str = "uniform",
        py_data_sampling_bins: int = 8,
        py_data_sampling_max_positions: int = 0,
        py_data_sampling_seed: int = 42,
        preference_data: bool = False,
        preference_context_type: str = "elo",
        elo_bucket_edges: tuple[int, ...] = (1200, 1600, 2000, 2400),
        elo_weight_slope: float = 0.001,
        elo_weight_intercept: float = 0.5,
        elo_weight_min: float = 0.1,
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

        # The C++ data loader needs a device.
        main_device = "cpu"
        if self.trainer:
            if self.trainer.strategy.root_device.type == "cuda":
                main_device = f"cuda:{self.trainer.strategy.root_device.index}"

        if self.hparams.preference_data:
            ds_cls = (
                preference_dataset.FixedRefBinaryDataset
                if Path(self.hparams.train).suffix == ".bin"
                else preference_dataset.FixedRefH5Dataset
            )
            self.train_ds = ds_cls(
                self.hparams.train,
                context_type=self.hparams.preference_context_type,
                elo_bucket_edges=self.hparams.elo_bucket_edges,
                elo_weight_slope=self.hparams.elo_weight_slope,
                elo_weight_intercept=self.hparams.elo_weight_intercept,
                elo_weight_min=self.hparams.elo_weight_min,
            )
            val_ds_cls = (
                preference_dataset.FixedRefBinaryDataset
                if Path(self.hparams.val).suffix == ".bin"
                else preference_dataset.FixedRefH5Dataset
            )
            self.val_ds = val_ds_cls(
                self.hparams.val,
                context_type=self.hparams.preference_context_type,
                elo_bucket_edges=self.hparams.elo_bucket_edges,
                elo_weight_slope=self.hparams.elo_weight_slope,
                elo_weight_intercept=self.hparams.elo_weight_intercept,
                elo_weight_min=self.hparams.elo_weight_min,
            )
        elif self.hparams.py_data:
            self.train_ds = nnue_bin_dataset.NNUEBinData(self.hparams.train, self.feature_set)
            self.val_ds = nnue_bin_dataset.NNUEBinData(self.hparams.val, self.feature_set)
        else:
            self.train_ds = nnue_dataset.FixedNumBatchesDataset(
                nnue_dataset.SparseBatchDataset(
                    self.hparams.features,
                    self.hparams.train,
                    self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    filtered=self.hparams.smart_fen_skipping,
                    random_fen_skipping=self.hparams.random_fen_skipping,
                    device=main_device,
                ),
                (self.hparams.epoch_size + self.hparams.batch_size - 1) // self.hparams.batch_size,
            )
            val_size = self.hparams.validation_size
            self.val_ds = nnue_dataset.FixedNumBatchesDataset(
                nnue_dataset.SparseBatchDataset(
                    self.hparams.features,
                    self.hparams.val,
                    self.hparams.batch_size,
                    filtered=self.hparams.smart_fen_skipping,
                    random_fen_skipping=self.hparams.random_fen_skipping,
                    device=main_device,
                ),
                (val_size + self.hparams.batch_size - 1) // self.hparams.batch_size,
            )

    def train_dataloader(self) -> DataLoader:
        if self.hparams.preference_data:
            train_workers = self.hparams.py_data_train_num_workers
            return DataLoader(
                self.train_ds,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=train_workers,
                persistent_workers=(train_workers > 0),
                collate_fn=preference_dataset.collate_fixed_ref_samples_for_training,
            )
        if self.hparams.py_data:
            sampler = nnue_bin_dataset.create_sampling_strategy(
                self.train_ds,
                mode=self.hparams.py_data_sampling_mode,
                num_bins=self.hparams.py_data_sampling_bins,
                max_positions=self.hparams.py_data_sampling_max_positions,
                seed=self.hparams.py_data_sampling_seed,
            )
            if sampler is not None:
                print(
                    "Using py_data sampler:",
                    self.hparams.py_data_sampling_mode,
                    f"(samples={len(sampler)})",
                )
            return DataLoader(
                self.train_ds,
                batch_size=self.hparams.batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=self.hparams.py_data_train_num_workers,
            )
        # The C++ loader already owns its own worker threads and prefetch queue.
        # Adding DataLoader worker processes on top multiplies memory usage.
        return DataLoader(self.train_ds, batch_size=None, batch_sampler=None, num_workers=0, persistent_workers=False)

    def val_dataloader(self) -> DataLoader:
        if self.hparams.preference_data:
            return DataLoader(
                self.val_ds,
                batch_size=self.hparams.py_data_val_batch_size,
                num_workers=0,
                collate_fn=preference_dataset.collate_fixed_ref_samples_for_training,
            )
        if self.hparams.py_data:
            return DataLoader(self.val_ds, batch_size=self.hparams.py_data_val_batch_size)
        return DataLoader(self.val_ds, batch_size=None, batch_sampler=None, num_workers=0, persistent_workers=False)


class PreferenceDataModule(pl.LightningDataModule):
    """Data module that uses the C++ DLL loader for preference data."""

    def __init__(
        self,
        train: str,
        val: str,
        features: str = "HalfKP",
        num_workers: int = 16,
        batch_size: int = 128,
        preference_context_type: str = "elo",
        elo_bucket_edges: tuple[int, ...] = (1200, 1600, 2000, 2400),
        elo_weight_slope: float = 0.001,
        elo_weight_intercept: float = 0.5,
        elo_weight_min: float = 0.1,
        threads: int = -1,
    ) -> None:
        super().__init__()
        if threads > 0:
            print(f"limiting torch to {threads} threads.")
            t_set_num_threads(threads)
        self.save_hyperparameters()
        self.feature_set = features_module.get_feature_set_from_name(self.hparams.features)

    def setup(self, stage: Optional[str] = None) -> None:
        main_device = "cpu"
        if self.trainer:
            if self.trainer.strategy.root_device.type == "cuda":
                main_device = f"cuda:{self.trainer.strategy.root_device.index}"

        h5_train = self.hparams.train
        h5_val = self.hparams.val

        # Convert H5 paths to .bin paths
        train_bin = str(Path(h5_train).with_suffix(".bin"))
        train_meta = train_bin.replace(".bin", "_meta.bin")
        val_bin = str(Path(h5_val).with_suffix(".bin"))
        val_meta = val_bin.replace(".bin", "_meta.bin")

        epoch_size = getattr(self.hparams, "epoch_size", 10_000_000)
        validation_size = getattr(self.hparams, "validation_size", 1_000_000)

        # Verify binary files exist; warn if missing
        if not Path(train_bin).exists():
            print(f"Warning: {train_bin} not found. Run: python export_preference_data.py {h5_train}")
            return
        if not Path(train_meta).exists():
            print(f"Warning: {train_meta} not found. Run: python export_preference_data.py {h5_train}")
            return
        if not Path(val_bin).exists():
            print(f"Warning: {val_bin} not found. Run: python export_preference_data.py {h5_val}")
            return
        if not Path(val_meta).exists():
            print(f"Warning: {val_meta} not found. Run: python export_preference_data.py {h5_val}")
            return

        self.train_ds = nnue_dataset.FixedNumBatchesDataset(
            nnue_dataset.SparseBatchDataset(
                self.hparams.features,
                train_bin,
                self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                filtered=False,
                random_fen_skipping=0,
                device=main_device,
            ),
            (epoch_size + self.hparams.batch_size - 1) // self.hparams.batch_size,
        )
        self.val_ds = nnue_dataset.FixedNumBatchesDataset(
            nnue_dataset.SparseBatchDataset(
                self.hparams.features,
                val_bin,
                self.hparams.batch_size,
                filtered=False,
                random_fen_skipping=0,
                device=main_device,
            ),
            (validation_size + self.hparams.batch_size - 1) // self.hparams.batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, "train_ds"):
            return DataLoader([], batch_size=None)
        return DataLoader(self.train_ds, batch_size=None, batch_sampler=None,
                          num_workers=0, persistent_workers=False)

    def val_dataloader(self) -> DataLoader:
        if not hasattr(self, "val_ds"):
            return DataLoader([], batch_size=None)
        return DataLoader(self.val_ds, batch_size=None, batch_sampler=None,
                          num_workers=0, persistent_workers=False)

class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("data.features", "model.features")


def _run_cli_subcommand(cli: LightningCLI) -> None:
    subcommand = getattr(cli, "subcommand", None) or "fit"
    if subcommand == "fit":
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        return
    if subcommand == "validate":
        cli.trainer.validate(cli.model, datamodule=cli.datamodule)
        return
    if subcommand == "test":
        cli.trainer.test(cli.model, datamodule=cli.datamodule)
        return
    if subcommand == "predict":
        cli.trainer.predict(cli.model, datamodule=cli.datamodule)
        return
    raise ValueError(f"Unsupported CLI subcommand: {subcommand}")


def main():
    argv = list(sys.argv[1:])
    requested_subcommand = "fit"
    if argv and argv[0] in {"fit", "validate", "test", "predict"}:
        requested_subcommand = argv.pop(0)

    # LightningCLI will add arguments for the model, datamodule, and trainer.
    # It will also handle seeding and checkpointing.
    # All model/data/trainer arguments are now passed through the command line
    # with dot notation, e.g., --model.lambda_ 0.5 or --data.batch_size 8192
    cli_kwargs = {"save_config_callback": None}
    try:
        cli = MyCLI(
            M.NNUE,
            NNUEDataModule,
            run=False,
            args=argv,
            parser_kwargs={"fit": {"default_config_files": ["config.yaml"]}},
            **cli_kwargs,
        )
    except TypeError as exc:
        if "default_config_files" not in str(exc):
            raise
        cli = MyCLI(M.NNUE, NNUEDataModule, run=False, args=argv, **cli_kwargs)
    cli.subcommand = requested_subcommand
    cli.trainer.callbacks.append(HParamsSnapshotCallback(cli.config))
    _run_cli_subcommand(cli)

if __name__ == "__main__":
    main()
