import io
import logging
import traceback
from typing import Self

import torch
from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph, TrainingBatch
from chemprop.nn import Aggregation, MessagePassing
from chemprop.nn.ffn import MLP
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from delta_data import RandomPairTrainBatch
from lightning import pytorch as pl
from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor, nn, optim

logger = logging.getLogger(__name__)


class Encoder(nn.Module, HyperparametersMixin):
    def __init__(
        self,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        output_dim: int = DEFAULT_HIDDEN_DIM,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str | nn.Module = "relu",
    ) -> None:
        super().__init__()
        # manually add criterion and output_transform to hparams to suppress lightning's warning
        # about double saving their state_dict values.
        ignore_list = ["activation"]
        self.save_hyperparameters(ignore=ignore_list)
        self.hparams["activation"] = activation
        # self.hparams["cls"] = self.__class__

        self.ffn = MLP.build(
            input_dim, output_dim, hidden_dim, n_layers, dropout, activation
        )

    def forward(self, Z: Tensor) -> Tensor:
        return self.ffn[:-1](Z)

    @property
    def input_dim(self):
        return self.ffn.input_dim

    @property
    def output_dim(self):
        return self.ffn.output_dim


class Interaction(torch.nn.Module, HyperparametersMixin):
    def __init__(self, ndims: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.interaction_matrix = torch.nn.Linear(ndims, ndims, bias=False)
        self.head_dropout = torch.nn.Dropout(dropout)

    def forward(self, head_emb: Tensor, tail_emb: Tensor):
        R = self.interaction_matrix.weight.unsqueeze(0)
        z = self.head_dropout(head_emb @ R) @ tail_emb.transpose(-2, -1)
        return z.squeeze()


class DeltaProp(pl.LightningModule):
    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        encoder: Encoder,
        interaction: Interaction,
        batch_norm: bool = False,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["X_d_transform", "message_passing", "agg", "encoder", "interaction"]
        )
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "encoder": encoder.hparams,
                "interaction": interaction.hparams,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.encoder = encoder
        self.interaction = interaction

        self.bn = (
            nn.BatchNorm1d(self.message_passing.output_dim)
            if batch_norm
            else nn.Identity()
        )
        self.X_d_transform = (
            X_d_transform if X_d_transform is not None else nn.Identity()
        )

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        self.loss_fn = nn.BCEWithLogitsLoss()

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)
        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        return self.encoder(self.fingerprint(bmg, V_d, X_d))

    def embed_simple_batch(self, batch: TrainingBatch):
        bmg, V_d, X_d, target, _, _, _ = batch
        Z = self.encoding(bmg, V_d, X_d)
        return dict(embeds=Z, targets=target)

    def bidirectional_interaction_loss(
        self, Z_left, Z_right, target_left, target_right
    ):
        # left to right loss
        lr_interaction = self.interaction(Z_left, Z_right).squeeze()
        lr_labels = target_left > target_right  # type: ignore
        lr_loss = self.loss_fn(lr_interaction, lr_labels.float())

        # right to left loss
        rl_interaction = self.interaction(Z_right, Z_left).squeeze()
        rl_labels = ~lr_labels  # type: ignore
        rl_loss = self.loss_fn(rl_interaction, rl_labels.float())

        delta = (lr_interaction.sigmoid() + rl_interaction.sigmoid() - 1.0) ** 2
        symm_loss = delta.sum(dim=-1).mean()

        return symm_loss, lr_loss, rl_loss

    def get_losses(self, batch: RandomPairTrainBatch):
        B, C = batch.B, batch.C

        bmg, V_d, X_d, target_anchor, _, _, _ = batch.anchor
        Z_anchor = self.encoding(bmg, V_d, X_d)

        bmg, V_d, X_d, target_exemplar, _, _, _ = batch.exemplar
        Z_exemplar = self.encoding(bmg, V_d, X_d)

        Z_anchor = Z_anchor.view(B, 1, -1)  # (B, d) -> (B, 1, d)
        Z_exemplar = Z_exemplar.view(B, C, -1)  # (B*C, d) -> (B, C, d)
        # Z_random = Z_random.view(B, C, -1)  # (B*C, d) -> (B, C, d)

        target_anchor = target_anchor.view(-1, 1)
        target_exemplar = target_exemplar.view(B, C)

        (exemplar_sym_loss, lr_exemplar_loss, rl_exemplar_loss) = (
            self.bidirectional_interaction_loss(
                Z_anchor, Z_exemplar, target_anchor, target_exemplar
            )
        )

        loss = (exemplar_sym_loss + lr_exemplar_loss + rl_exemplar_loss) / 3

        return loss, (
            exemplar_sym_loss,
            lr_exemplar_loss,
            rl_exemplar_loss,
        )

    def training_step(self, batch: RandomPairTrainBatch, batch_idx):  # type: ignore
        (
            loss,
            (
                exemplar_sym_loss,
                lr_exemplar_loss,
                rl_exemplar_loss,
            ),
        ) = self.get_losses(batch)

        self.log(
            "train_exemplar_sym_loss",
            exemplar_sym_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log(
            "train_lr_exemplar_loss",
            lr_exemplar_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log(
            "train_rl_exemplar_loss",
            rl_exemplar_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log("train_loss", loss, batch_size=batch.B, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: RandomPairTrainBatch, batch_idx):  # type: ignore
        (
            loss,
            (
                exemplar_sym_loss,
                lr_exemplar_loss,
                rl_exemplar_loss,
            ),
        ) = self.get_losses(batch)

        self.log(
            "val_exemplar_sym_loss",
            exemplar_sym_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log(
            "val_lr_exemplar_loss",
            lr_exemplar_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log(
            "val_rl_exemplar_loss",
            rl_exemplar_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log("val_loss", loss, batch_size=batch.B, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch

        if self.trainer.max_epochs == -1:
            logger.warning(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            assert self.trainer.max_epochs is not None
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt,
            int(warmup_steps),
            int(cooldown_steps),
            self.init_lr,
            self.max_lr,
            self.final_lr,
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    @classmethod
    def _load(cls, path, map_location, **submodules):
        try:
            d = torch.load(path, map_location, weights_only=False)
        except AttributeError:
            logger.error(f"{traceback.format_exc()}")

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(
                f"Could not find hyper parameters and/or state dict in {path}."
            )

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "encoder", "interaction")
            if key not in submodules
        }

        return submodules, state_dict, hparams

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        **kwargs,
    ) -> Self:
        submodules = {
            k: v
            for k, v in kwargs.items()
            if k in ["message_passing", "agg", "predictor"]
        }
        submodules, state_dict, hparams = cls._load(
            checkpoint_path, map_location, **submodules
        )
        kwargs.update(submodules)

        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d["state_dict"] = state_dict
        d["hyper_parameters"] = hparams
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(
            buffer, map_location, hparams_file, strict, **kwargs
        )

    @classmethod
    def load_from_file(
        cls, model_path, map_location=None, strict=True, **submodules
    ) -> Self:
        submodules, state_dict, hparams = cls._load(
            model_path, map_location, **submodules
        )
        hparams.update(submodules)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model
