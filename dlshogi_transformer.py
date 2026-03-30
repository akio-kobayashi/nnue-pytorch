from collections.abc import Callable, Iterator
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer


Batch = tuple[Tensor, ...]
TensorDict = dict[str, Tensor]


class DLShogiTransformer(pl.LightningModule):
    def __init__(
        self,
        lr: list[float] | None = None,
        lambda_: list[float] | None = None,
        label_smoothing_eps: float = 0.0,
        num_batches_warmup: int = 4000,
        score_scaling: float = 361.0,
        ply_begin_threshold: float = 100.0,
        ply_end_threshold: float = 120.0,
        teacher_temperature: float = 1.0,
        entropy_coef: float = 1.0,
        outcome_pos_weight: float = 1.0,
        optimizer: str = "adamw",
        weight_decay: float = 0.01,
        momentum: float = 0.9,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 256,
        ema_enabled: bool = False,
        ema_decay: float = 0.9995,
        ema_update_every: int = 1,
        ema_start_step: int = 1000,
        train_loss_ema_beta: float = 0.98,
    ):
        super().__init__()
        if lr is None:
            lr = [3.0e-4]
        if lambda_ is None:
            lambda_ = [1.0]
        self.save_hyperparameters()

        self.NNUE_TO_SCORE = 600.0
        self.EPSILON = 1e-12
        self.FEATURES1_NUM = 62
        self.FEATURES2_NUM = 57
        self.NUM_SQUARES = 81

        self.lr = lr
        self.lambda_ = lambda_
        self.label_smoothing_eps = label_smoothing_eps
        self.num_batches_warmup = num_batches_warmup
        self.score_scaling = score_scaling
        self.ply_begin_threshold = ply_begin_threshold
        self.ply_end_threshold = ply_end_threshold
        self.teacher_temperature = max(float(teacher_temperature), self.EPSILON)
        self.entropy_coef = float(entropy_coef)
        self.outcome_pos_weight = max(float(outcome_pos_weight), self.EPSILON)
        self.optimizer_name = optimizer.lower()
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.validation_step_outputs = []
        self.ema_enabled = ema_enabled
        self.ema_decay = ema_decay
        self.ema_update_every = max(1, int(ema_update_every))
        self.ema_start_step = max(0, int(ema_start_step))
        self.train_loss_ema_beta = min(max(float(train_loss_ema_beta), 0.0), 0.9999)
        self._ema_state: TensorDict = {}
        self._ema_backup: TensorDict | None = None
        self._train_loss_ema: Tensor | None = None
        self.warmup_start_global_step = 0

        self.square_proj = nn.Linear(self.FEATURES1_NUM, d_model)
        self.global_proj = nn.Linear(self.FEATURES2_NUM, d_model)
        self.square_pos = nn.Parameter(torch.zeros(1, self.NUM_SQUARES, d_model))
        self.global_token_bias = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pre_head_norm = nn.LayerNorm(2 * d_model)
        self.pre_head = nn.Sequential(
            nn.Linear(2 * d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(mlp_hidden_dim, 1)

    def _forward_hidden(self, features1: Tensor, features2: Tensor) -> Tensor:
        square_tokens = features1.transpose(1, 2)
        square_tokens = self.square_proj(square_tokens) + self.square_pos

        global_features = features2[:, :, 0]
        global_token = self.global_proj(global_features).unsqueeze(1) + self.global_token_bias

        x = torch.cat([global_token, square_tokens], dim=1)
        x = self.encoder(x)
        summary = torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=1)
        summary = self.pre_head_norm(summary)
        return self.pre_head(summary)

    def forward(self, features1: Tensor, features2: Tensor) -> Tensor:
        return self.output(self._forward_hidden(features1, features2))

    def _compute_lambda(self, ply: Tensor) -> Tensor | float:
        lambda_base = self.lambda_[0]
        if lambda_base >= 0.0:
            return lambda_base
        lambda_ = (self.ply_end_threshold - ply) / (
            self.ply_end_threshold - self.ply_begin_threshold
        )
        return torch.clamp(lambda_, 0.0, 1.0)

    def _compute_loss_terms(
        self,
        q: Tensor,
        outcome: Tensor,
        score: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        t = outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
        p = (score / (self.score_scaling * self.teacher_temperature)).sigmoid()
        teacher_entropy = -(p * (p + self.EPSILON).log() + (1.0 - p) * (1.0 - p + self.EPSILON).log())
        outcome_entropy = -(t * (t + self.EPSILON).log() + (1.0 - t) * (1.0 - t + self.EPSILON).log())
        teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
        outcome_loss = -(
            self.outcome_pos_weight * t * F.logsigmoid(q)
            + (1.0 - t) * F.logsigmoid(-q)
        )
        return teacher_entropy, outcome_entropy, teacher_loss, outcome_loss

    def step_(self, batch: Batch, batch_idx: int, loss_type: str) -> Tensor:
        features1, features2, outcome, score, ply = batch
        hidden = self._forward_hidden(features1, features2)
        q = self.output(hidden) * self.NNUE_TO_SCORE / self.score_scaling
        teacher_entropy, outcome_entropy, teacher_loss, outcome_loss = self._compute_loss_terms(
            q, outcome, score
        )
        lambda_ = self._compute_lambda(ply)
        result = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
        entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
        loss = result.mean() - self.entropy_coef * entropy.mean()
        batch_size = features1.shape[0]

        if loss_type == "train_loss":
            self.log("train_loss_step", loss, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log(
                "train_teacher_loss",
                teacher_loss.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "train_outcome_loss",
                outcome_loss.mean(),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

            detached_loss = loss.detach()
            if self._train_loss_ema is None:
                self._train_loss_ema = detached_loss
            else:
                beta = self.train_loss_ema_beta
                self._train_loss_ema = beta * self._train_loss_ema + (1.0 - beta) * detached_loss
            self.log(
                "train_loss_ema",
                self._train_loss_ema,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
        else:
            self.log(loss_type, loss, on_step=False, on_epoch=True, prog_bar=(loss_type == "val_loss"), batch_size=batch_size)
        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        loss = self.step_(batch, batch_idx, "val_loss")
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        self.step_(batch, batch_idx, "test_loss")

    def _iter_ema_parameters(self) -> Iterator[tuple[str, Tensor]]:
        for name, param in self.named_parameters():
            if param.requires_grad and torch.is_floating_point(param):
                yield name, param

    def _initialize_ema_state(self) -> None:
        if self._ema_state:
            return
        self._ema_state = {
            name: param.detach().clone()
            for name, param in self._iter_ema_parameters()
        }

    def _update_ema_state(self) -> None:
        self._initialize_ema_state()
        decay = float(self.ema_decay)
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for name, param in self._iter_ema_parameters():
                self._ema_state[name].mul_(decay).add_(param.detach(), alpha=one_minus_decay)

    def apply_ema_weights(self) -> bool:
        if not self._ema_state:
            return False
        if self._ema_backup is not None:
            return True
        self._ema_backup = {}
        with torch.no_grad():
            for name, param in self._iter_ema_parameters():
                ema_weight = self._ema_state.get(name)
                if ema_weight is None:
                    continue
                self._ema_backup[name] = param.detach().clone()
                param.copy_(ema_weight.to(device=param.device, dtype=param.dtype))
        return True

    def restore_original_weights(self) -> None:
        if self._ema_backup is None:
            return
        with torch.no_grad():
            for name, param in self._iter_ema_parameters():
                original = self._ema_backup.get(name)
                if original is None:
                    continue
                param.copy_(original.to(device=param.device, dtype=param.dtype))
        self._ema_backup = None

    def on_fit_start(self) -> None:
        if self.ema_enabled:
            self._initialize_ema_state()

    def on_train_batch_end(self, outputs: Any, batch: Batch, batch_idx: int) -> None:
        if not self.ema_enabled:
            return
        global_step = self.trainer.global_step
        if global_step < self.ema_start_step:
            return
        if (global_step - self.ema_start_step) % self.ema_update_every != 0:
            return
        self._update_ema_state()

    def on_validation_epoch_start(self) -> None:
        if self.ema_enabled:
            self.apply_ema_weights()

    def on_validation_epoch_end(self) -> None:
        try:
            if not self.validation_step_outputs:
                return
            self.validation_step_outputs.clear()
        finally:
            self.restore_original_weights()

    def on_test_epoch_start(self) -> None:
        if self.ema_enabled:
            self.apply_ema_weights()

    def on_test_epoch_end(self) -> None:
        self.restore_original_weights()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self._ema_state:
            checkpoint["ema_state"] = {
                name: tensor.detach().cpu()
                for name, tensor in self._ema_state.items()
            }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        loaded_ema = checkpoint.get("ema_state")
        if isinstance(loaded_ema, dict):
            self._ema_state = {
                name: tensor.clone()
                for name, tensor in loaded_ema.items()
                if isinstance(tensor, Tensor)
            }

    def _apply_learning_rate(self, optimizer: Optimizer) -> None:
        if self.trainer.global_step - self.warmup_start_global_step < self.num_batches_warmup:
            warmup_scale = min(
                1.0,
                float(self.trainer.global_step - self.warmup_start_global_step + 1)
                / self.num_batches_warmup,
            )
        else:
            warmup_scale = 1.0

        for pg in optimizer.param_groups:
            pg["lr"] = self.lr[0] * warmup_scale
            self.log("lr", pg["lr"])

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_closure: Callable[[], Any],
    ) -> None:
        self._apply_learning_rate(optimizer)
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self) -> Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                self.parameters(),
                lr=self.lr[0],
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr[0],
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
