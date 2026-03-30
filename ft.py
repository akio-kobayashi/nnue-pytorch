import model as M
import features as features_module
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


def _infer_features_from_state_dict(state_dict) -> str:
  input_dim = state_dict["input.weight"].shape[1]
  for feature_name in features_module.get_available_feature_blocks_names():
    feature_set = features_module.get_feature_set_from_name(feature_name)
    if feature_set.num_real_features == input_dim or feature_set.num_features == input_dim:
      return feature_name
  raise ValueError(f"Could not infer feature set from input dimension {input_dim}")


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
    if not base_ckpt:
      raise ValueError("base_ckpt must be provided for fine-tuning")

    adapter_config = {
        "input_adapter": kwargs.pop("input_adapter", "none"),
        "input_adapter_rank": kwargs.pop("input_adapter_rank", 8),
        "input_adapter_alpha": kwargs.pop("input_adapter_alpha", 1.0),
        "input_adapter_init_std": kwargs.pop("input_adapter_init_std", 1e-3),
        "freeze_base_input": kwargs.pop("freeze_base_input", False),
    }

    state_dict, checkpoint = _load_checkpoint_state(base_ckpt)
    checkpoint_features = _infer_features_from_state_dict(state_dict)
    super().__init__(features=checkpoint_features, input_adapter="none", **kwargs)
    self.load_state_dict(state_dict, strict=True)
    if checkpoint and hasattr(self, "on_load_checkpoint"):
      self.on_load_checkpoint(checkpoint)

    if features != checkpoint_features:
      self.set_feature_set(features_module.get_feature_set_from_name(features))

    self.configure_input_adapter(**adapter_config)

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
      self.freeze_input_adapter_parameters()
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
