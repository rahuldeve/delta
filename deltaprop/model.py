import io
import logging
import traceback
from typing import Self

import torch
from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph, TrainingBatch
from chemprop.nn import Aggregation, BondMessagePassing, MessagePassing, MeanAggregation
from chemprop.nn.ffn import MLP
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from lightning import pytorch as pl
from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn, optim

from deltaprop.data import RandomPairTrainBatch

logger = logging.getLogger(__name__)


class Encoder(nn.Module, HyperparametersMixin):
    def __init__(
        self,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
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
        self.hparams["cls"] = self.__class__

        self.ffn = MLP.build(
            input_dim, output_dim, hidden_dim, n_layers, dropout, activation
        )

    def forward(self, Z: Tensor) -> Tensor:
        return self.ffn(Z)

    @property
    def input_dim(self):
        return self.ffn.input_dim

    @property
    def output_dim(self):
        return self.ffn.output_dim
    

class Stack(torch.nn.Module, HyperparametersMixin):
    def __init__(
        self, 
        message_passing: MessagePassing,
        agg: Aggregation,
        encoder: Encoder,
        batch_norm: bool = False,
        # X_d_transform: ScaleTransform | None = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["message_passing", "agg", "encoder"]
        )

        self.hparams["cls"] = self.__class__
        # self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "encoder": encoder.hparams,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.encoder = encoder

        self.bn = (
            nn.BatchNorm1d(self.message_passing.output_dim)
            if batch_norm
            else nn.Identity()
        )

        self.ln = nn.LayerNorm(self.encoder.input_dim)

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        Z = self.encoder(
            H if X_d is None 
            else self.ln(torch.cat((H, X_d), dim=1))
        )

        return Z + H


class Interaction(torch.nn.Module, HyperparametersMixin):
    def __init__(self, ndims: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        # self.interaction_matrix = torch.nn.Linear(ndims, ndims, bias=False)
        self.head_dropout = torch.nn.Dropout(dropout)

    def forward(self, head_emb: Tensor, tail_emb: Tensor):
        # R = self.interaction_matrix.weight.unsqueeze(0)
        z = head_emb @ tail_emb.transpose(-2, -1)
        return z.squeeze()


class DeltaProp(pl.LightningModule):
    def __init__(
        self,
        left_encoder: Stack,
        right_encoder: Stack,
        interaction: Interaction,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["X_d_transform", "left_encoder", "right_encoder", "interaction"]
        )
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update(
            {
                "left_encoder": left_encoder.hparams,
                "right_encoder": right_encoder.hparams,
                "interaction": interaction.hparams
            }
        )

        self.left_encoder = left_encoder
        self.right_encoder = right_encoder
        self.interaction = interaction

        self.X_d_transform = (
            X_d_transform if X_d_transform is not None else nn.Identity()
        )

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        self.loss_fn = nn.BCEWithLogitsLoss()

    def left_encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        return self.left_encoder(bmg, V_d, self.X_d_transform(X_d))
    
    def right_encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        return self.right_encoder(bmg, V_d, self.X_d_transform(X_d))

    def left_embed_simple_batch(self, batch: TrainingBatch):
        bmg, V_d, X_d, target, _, _, _ = batch
        Z = self.left_encoding(bmg, V_d, X_d)
        return dict(embeds=Z, targets=target)
    
    def right_embed_simple_batch(self, batch: TrainingBatch):
        bmg, V_d, X_d, target, _, _, _ = batch
        Z = self.right_encoding(bmg, V_d, X_d)
        return dict(embeds=Z, targets=target)

    def bidirectional_interaction_loss(self, batch):   
        B, C = batch.B, batch.C
        bmg_anchor, V_d_anchor, X_d_anchor, target_anchor, _, _, _ = batch.anchor
        bmg_candidates, V_d_candidates, X_d_candidates, target_candidates, _, _, _ = batch.candidates
        
        # anchor to candidate loss
        Z_left = self.left_encoding(bmg_anchor, V_d_anchor, X_d_anchor).view(B, 1, -1)  # (B, d) -> (B, 1, d)
        Z_right = self.right_encoding(bmg_candidates, V_d_candidates, X_d_candidates).view(B, C, -1)  # (B*C, d) -> (B, C, d)

        target_left = target_anchor.view(-1, 1) # type: ignore
        target_right = target_candidates.view(B, C) # type: ignore

        lr_interaction = self.interaction(Z_left, Z_right).squeeze()
        lr_labels = (target_left > target_right).squeeze()  # type: ignore
        lr_loss = self.loss_fn(lr_interaction, lr_labels.float())


        # candidate to anchor loss
        Z_left = self.left_encoding(bmg_candidates, V_d_candidates, X_d_candidates).view(B, C, -1)  # (B*C, d) -> (B, C, d)
        Z_right = self.right_encoding(bmg_anchor, V_d_anchor, X_d_anchor).view(B, 1, -1)  # (B, d) -> (B, 1, d)

        target_left = target_candidates.view(B, C) # type: ignore
        target_right = target_anchor.view(-1, 1) # type: ignore

        rl_interaction = self.interaction(Z_right, Z_left).squeeze()
        rl_labels = ~lr_labels  # type: ignore
        rl_loss = self.loss_fn(rl_interaction, rl_labels.float())

        delta = (lr_interaction.sigmoid() + rl_interaction.sigmoid() - 1.0) ** 2
        symm_loss = delta.sum(dim=-1).mean()

        return symm_loss, lr_loss, rl_loss

    def get_losses(self, batch: RandomPairTrainBatch):

        (sym_loss, lr_loss, rl_loss) = self.bidirectional_interaction_loss(batch)

        loss = (sym_loss + lr_loss + rl_loss) / 3
        return loss, (sym_loss, lr_loss, rl_loss)

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.left_encoder.message_passing.V_d_transform.train() # type: ignore
        self.left_encoder.message_passing.graph_transform.train() # type: ignore
        self.right_encoder.message_passing.V_d_transform.train() # type: ignore
        self.right_encoder.message_passing.graph_transform.train() # type: ignore
        self.X_d_transform.train()

    def training_step(self, batch: RandomPairTrainBatch, batch_idx):  # type: ignore
        loss, (sym_loss, lr_loss, rl_loss) = self.get_losses(batch)

        self.log(
            "train_sym_loss",
            sym_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log(
            "train_lr_loss",
            lr_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log(
            "train_rl_loss",
            rl_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log("train_loss", loss, batch_size=batch.B, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: RandomPairTrainBatch, batch_idx):  # type: ignore
        loss, (sym_loss, lr_loss, rl_loss) = self.get_losses(batch)

        self.log(
            "val_sym_loss",
            sym_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log(
            "val_lr_loss",
            lr_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log(
            "val_rl_loss",
            rl_loss,
            batch_size=batch.B,
            on_epoch=True,
            enable_graph=True,
        )

        self.log("val_loss", loss, batch_size=batch.B, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self): # type: ignore
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
            hparams = d["hyper_parameters"] # type: ignore
            state_dict = d["state_dict"] # type: ignore
        except KeyError:
            raise KeyError(
                f"Could not find hyper parameters and/or state dict in {path}."
            )

        left_encoder_submodules = {
            key: hparams["left_encoder"][key].pop("cls")(**hparams["left_encoder"][key])
            for key in ("message_passing", "agg", "encoder")
        }

        left_encoder = {
            "left_encoder": hparams["left_encoder"].pop("cls")(**left_encoder_submodules)
        }

        right_encoder_submodules = {
            key: hparams["right_encoder"][key].pop("cls")(**hparams["right_encoder"][key])
            for key in ("message_passing", "agg", "encoder")
        }

        right_encoder = {
            "right_encoder": hparams["right_encoder"].pop("cls")(**right_encoder_submodules)
        }

        interaction = {
            "interaction": hparams["interaction"].pop("cls")(**hparams["interaction"])
        }

        submodules = {
            k:v for k,v in (left_encoder | right_encoder | interaction).items()
            if k not in submodules
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
            if k in ["left_encoder", "right_encoder", "interaction"]
        }
        submodules, state_dict, hparams = cls._load(
            checkpoint_path, map_location, **submodules
        )
        kwargs.update(submodules)

        d = torch.load(checkpoint_path, map_location, weights_only=False) # type: ignore
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


def build_model(config, X_d_scaler: StandardScaler | None) -> DeltaProp:
    depth = config["depth"]
    ffn_hidden_dim = config["ffn_hidden_dim"]
    ffn_num_layers = config["ffn_num_layers"]
    message_hidden_dim = config["message_hidden_dim"]
    batch_norm = config["batch_norm"]
    encoder_dropout = config["encoder_dropout"]
    interaction_dropout = config["interaction_dropout"]

    if X_d_scaler is not None:
        X_d_transform = ScaleTransform.from_standard_scaler(X_d_scaler)
        num_mol_feats = X_d_scaler.n_features_in_
    else:
        X_d_transform = None
        num_mol_feats = 0

    mp = BondMessagePassing(d_h=message_hidden_dim, depth=depth) # type: ignore
    agg = MeanAggregation()
    ffn_dims = mp.output_dim + num_mol_feats
    encoder = Encoder(
        input_dim=ffn_dims,
        hidden_dim=ffn_hidden_dim,
        output_dim=message_hidden_dim,
        n_layers=ffn_num_layers,
        activation=torch.nn.PReLU(),
        dropout=encoder_dropout,
    )

    left_encoder = Stack(message_passing=mp, agg=agg, encoder=encoder)

    mp = BondMessagePassing(d_h=message_hidden_dim, depth=depth) # type: ignore
    agg = MeanAggregation()
    ffn_dims = mp.output_dim + num_mol_feats
    encoder = Encoder(
        input_dim=ffn_dims,
        hidden_dim=ffn_hidden_dim,
        output_dim=message_hidden_dim,
        n_layers=ffn_num_layers,
        activation=torch.nn.PReLU(),
        dropout=encoder_dropout,
    )
    
    right_encoder = Stack(message_passing=mp, agg=agg, encoder=encoder)

    interaction = Interaction(encoder.output_dim, dropout=interaction_dropout)

    X_d_transform = (
        ScaleTransform.from_standard_scaler(X_d_scaler)
        if X_d_scaler is not None
        else None
    )

    model = DeltaProp(left_encoder=left_encoder, right_encoder=right_encoder, interaction=interaction, X_d_transform=X_d_transform)

    return model
