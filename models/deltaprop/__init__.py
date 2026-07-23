import random
import tempfile
from typing import Self

import lightning as L
import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDataset
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import (
    BondMessagePassing,
    NormAggregation,
    ScaleTransform,
)
from ghostml import optimize_threshold_from_predictions
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.preprocessing import StandardScaler

from config import TrainConfig
from data import DSThreshold
from models.abc import PreparedDatasetSplit, RefModel
from models.config import DeltapropConfig
from models.deltaprop.data import (
    DeltaMoleculeDatapoint,
    DeltaMoleculeDataset,
    RandomPairDataModule,
)
from models.deltaprop.model import DeltaProp, Encoder, Interaction, TiedClassifier
from models.deltaprop.utils import classifier_probs


def get_molecule_datapoint(row):
    feat_entry_names = [f for f in row.index if f.startswith("feat")]
    if len(feat_entry_names) > 0:
        feat_array = pd.to_numeric(row[feat_entry_names], errors="coerce")
    else:
        feat_array = None

    return DeltaMoleculeDatapoint(
        mol=row["mol"],
        # `y` (the continuous target) is pegged to reg_Y and drives the ranking
        # head; `bin_y` (the binary label) feeds the classification head.
        y=np.array([row["cont_target"]]),
        bin_y=np.array([row["bin_target"]], dtype=float),
        x_d=feat_array,
    )


# ref: https://docs.pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DeltapropRef(RefModel[DeltapropConfig]):
    def __init__(self, model: DeltaProp) -> None:
        self.model = model

    @staticmethod
    def prepare_splits(*, train_df, val_df, test_df):
        train_dps = train_df.apply(get_molecule_datapoint, axis=1).tolist()
        val_dps = val_df.apply(get_molecule_datapoint, axis=1).tolist()
        test_dps = test_df.apply(get_molecule_datapoint, axis=1).tolist()

        featurizer = SimpleMoleculeMolGraphFeaturizer()
        train_mol_dataset = DeltaMoleculeDataset(train_dps, featurizer=featurizer)
        val_mol_dataset = DeltaMoleculeDataset(val_dps, featurizer=featurizer)
        test_mol_dataset = DeltaMoleculeDataset(test_dps, featurizer=featurizer)

        X_d_scaler = train_mol_dataset.normalize_inputs("X_d")
        val_mol_dataset.normalize_inputs("X_d", X_d_scaler)

        train_mol_dataset.cache = True
        val_mol_dataset.cache = True

        return PreparedDatasetSplit(
            train_split=train_mol_dataset,
            val_split=val_mol_dataset,
            test_split=test_mol_dataset,
            extras=dict(X_d_scaler=X_d_scaler),
        )

    @classmethod
    def build(
        cls,
        *,
        model_config: DeltapropConfig,
        X_d_scaler: StandardScaler | None,
        **kwargs,
    ) -> "DeltapropRef":
        if X_d_scaler is not None:
            X_d_transform = ScaleTransform.from_standard_scaler(X_d_scaler)
            num_mol_feats = X_d_scaler.n_features_in_
        else:
            X_d_transform = None
            num_mol_feats = 0

        if model_config.use_chameleon_mp:
            chemeleon_mp = torch.load("./chemeleon_mp.pt", weights_only=True)
            mp = BondMessagePassing(**chemeleon_mp["hyper_parameters"])  # type: ignore
            mp.load_state_dict(chemeleon_mp["state_dict"])
        else:
            mp = BondMessagePassing(
                d_h=model_config.mp_d_h,
                depth=model_config.mp_depth,
                dropout=model_config.mp_dropout,
            )  # type: ignore

        agg = NormAggregation()
        ffn_dims = mp.output_dim + num_mol_feats
        encoder = Encoder(
            input_dim=ffn_dims,
            hidden_dim=model_config.encoder_hidden_dim,
            output_dim=model_config.encoder_output_dim,
            n_layers=model_config.encoder_n_layers,
            dropout=model_config.encoder_dropout,
            activation="elu",
        )
        interaction = Interaction(
            encoder.output_dim, dropout=model_config.interaction_dropout
        )
        # Hardcoded to the tied head for now: the binary logit is an affine map of the
        # Davidson strength λ, detached so the classification loss trains only
        # `scale`/`bias` and the encoder stays shaped by the ranking objective alone.
        # NOTE: `positive_is_greater` defaults to True, which is the wrong sign for the
        # LT datasets (DB_MALARIA, DB_HEPG2) — `scale` has to learn through zero there.
        classifier = TiedClassifier(detach=True)

        X_d_transform = (
            ScaleTransform.from_standard_scaler(X_d_scaler)
            if X_d_scaler is not None
            else None
        )
        model = DeltaProp(
            mp,
            agg,
            encoder,
            interaction,
            classifier,
            X_d_transform=X_d_transform,
            batch_norm=model_config.batch_norm,
            ranking_loss_weight=model_config.ranking_loss_weight,
        )

        return DeltapropRef(model)

    def train_func(
        self,
        *,
        train_split: MoleculeDataset,
        val_split: MoleculeDataset,
        train_config: TrainConfig,
        df_classification_threshold: DSThreshold,
        model_config: DeltapropConfig,
        **kwargs,
    ) -> Self:
        datamodule = RandomPairDataModule(
            train_mol_ds=train_split,
            val_mol_ds=val_split,
            batch_size=train_config.batch_size,
            n_candidates=model_config.candidate_size,
            frac_hard=model_config.frac_hard,
            seed=train_config.random_seed,
        )

        with tempfile.TemporaryDirectory() as ckpt_dir:
            trainer = L.Trainer(
                logger=None,
                enable_checkpointing=True,
                enable_progress_bar=True,
                accelerator="auto",
                devices=1,
                max_epochs=train_config.max_epochs,
                num_sanity_val_steps=0,
                reload_dataloaders_every_n_epochs=2,
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        verbose=True,
                        patience=train_config.early_stopping_patience,
                    ),
                    ModelCheckpoint(
                        dirpath=ckpt_dir,
                        monitor="val_loss",
                        mode="min",
                        save_top_k=1,
                    ),
                ],
            )

            trainer.fit(self.model, datamodule=datamodule)
            self.model = DeltaProp.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,  # type: ignore
                weights_only=False,
            )
        return self

    def tune_binary_classification_threshold(
        self,
        *,
        train_split: MoleculeDataset,
        train_labels: np.typing.NDArray[np.bool],
        val_split: MoleculeDataset,
        val_labels: np.typing.NDArray[np.bool],
        df_classification_threshold: DSThreshold,
        train_config: TrainConfig,
        **kwargs,
    ) -> float:
        model = self.model
        model.eval()

        # Probabilities come straight from the classification head; val X_d is
        # pre-scaled in prepare_splits, so scale_X_d stays False (the default).
        pred_probs = classifier_probs(val_split, model)

        # A direct sigmoid classifier's optimal threshold can sit above 0.5, so scan
        # the full (0.05, 0.95) range rather than the ~0.5-centered class-mean range.
        thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
        optimal_threshold = optimize_threshold_from_predictions(
            labels=val_labels,
            probs=pred_probs,
            thresholds=thresholds,
            random_seed=train_config.random_seed,
        )

        return optimal_threshold

    def predict_func(
        self,
        *,
        binary_classification_threshold: float,
        df_classification_threshold: DSThreshold,
        train_split: MoleculeDataset,
        train_labels: np.typing.NDArray[np.bool],
        test_split: MoleculeDataset,
        split_X_d_prescaled: bool = False,
        **kwargs,
    ):
        model = self.model
        model.eval()

        # Probabilities come straight from the classification head. `test_split` is
        # raw for the test set but pre-scaled for the val set; `split_X_d_prescaled`
        # avoids double-scaling its features.
        pred_probs = classifier_probs(
            test_split, model, scale_X_d=not split_X_d_prescaled
        )

        preds = (pred_probs >= binary_classification_threshold).astype(float)

        return pred_probs, preds
