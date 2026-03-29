import model as M
import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningCLI

from train import NNUEDataModule


def _load_checkpoint_state(base_ckpt: str):
  checkpoint = torch.load(base_ckpt, map_location="cpu")
  if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    return checkpoint["state_dict"], checkpoint
  if isinstance(checkpoint, dict):
    return checkpoint, {}
  if hasattr(checkpoint, "state_dict"):
    return checkpoint.state_dict(), {}
  raise TypeError(f"Unsupported checkpoint format for {base_ckpt}")


class FineTuningNNUE(M.NNUE):
  def __init__(
      self,
      base_ckpt: str,
      features: str,
      use_ema_weights: bool = False,
      freeze_input: bool = False,
      freeze_l1: bool = False,
      freeze_l2: bool = False,
      freeze_output: bool = False,
      freeze_aux_heads: bool = True,
      **kwargs,
  ):
    super().__init__(features=features, **kwargs)
    if not base_ckpt:
      raise ValueError("base_ckpt must be provided for fine-tuning")

    state_dict, checkpoint = _load_checkpoint_state(base_ckpt)
    self.load_state_dict(state_dict, strict=True)
    if checkpoint and hasattr(self, "on_load_checkpoint"):
      self.on_load_checkpoint(checkpoint)

    if use_ema_weights:
      if not hasattr(self, "apply_ema_weights") or not self.apply_ema_weights():
        raise RuntimeError("Requested use_ema_weights=True but no EMA weights were found in the checkpoint.")
      # Fine-tuning should start from the EMA parameters themselves, not bounce back
      # to the pre-EMA parameters during validation hooks.
      self._ema_backup = None

    self._apply_freeze(
        freeze_input=freeze_input,
        freeze_l1=freeze_l1,
        freeze_l2=freeze_l2,
        freeze_output=freeze_output,
        freeze_aux_heads=freeze_aux_heads,
    )

  def _freeze_module(self, module):
    for param in module.parameters():
      param.requires_grad = False

  def _apply_freeze(
      self,
      freeze_input: bool,
      freeze_l1: bool,
      freeze_l2: bool,
      freeze_output: bool,
      freeze_aux_heads: bool,
  ) -> None:
    if freeze_input:
      self._freeze_module(self.input)
    if freeze_l1:
      self._freeze_module(self.l1)
    if freeze_l2:
      self._freeze_module(self.l2)
    if freeze_output:
      self._freeze_module(self.output)
    if freeze_aux_heads:
      self._freeze_module(self.king_zone_head)
      self._freeze_module(self.major_safety_head)


class FineTuneCLI(LightningCLI):
  def add_arguments_to_parser(self, parser) -> None:
    parser.link_arguments("data.features", "model.features")


def main():
  cli_kwargs = {"save_config_callback": None}
  try:
    FineTuneCLI(
        FineTuningNNUE,
        NNUEDataModule,
        parser_kwargs={"fit": {"default_config_files": ["ft.yaml"]}},
        **cli_kwargs,
    )
  except TypeError as exc:
    if "default_config_files" not in str(exc):
      raise
    FineTuneCLI(FineTuningNNUE, NNUEDataModule, **cli_kwargs)


if __name__ == "__main__":
  main()
