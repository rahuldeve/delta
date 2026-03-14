import random
from typing import Self

import lightning as L
import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.data.dataloader import collate_batch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import (
    BinaryClassificationFFN,
    BondMessagePassing,
    NormAggregation,
    ScaleTransform,
    metrics,
)
from ghostml import optimize_threshold_from_predictions
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import move_data_to_device
from sklearn.preprocessing import StandardScaler

from config import TrainConfig
from misc import set_seeds
from models.abc import PreparedDatasetSplit, RefModel
from models.config import BaselineConfig


def get_molecule_datapoint(row):
    feat_entry_names = [f for f in row.index if f.startswith("feat")]
    if len(feat_entry_names) > 0:
        feat_array = pd.to_numeric(row[feat_entry_names], errors="coerce")
    else:
        feat_array = None

    return MoleculeDatapoint(
        mol=row["mol"], y=np.array([row["bin_target"]]), x_d=feat_array
    )


# ref: https://docs.pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def get_pred_probs(model: MPNN, mol_ds: MoleculeDataset, scale_X_d=False):
    model.eval()
    if not scale_X_d:
        model.X_d_transform.train()

    dl = torch.utils.data.DataLoader(
        mol_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_batch,
    )

    pred_probs = []
    for batch in dl:
        batch = move_data_to_device(batch, model.device)
        bmg, V_d, X_d, _, _, _, _ = batch
        pred_probs.append(model(bmg, V_d, X_d))

    pred_probs = torch.cat(pred_probs).squeeze().cpu().numpy()
    return pred_probs


class ChempropRef(RefModel[BaselineConfig]):
    def __init__(self, model: MPNN) -> None:
        self.model = model

    @staticmethod
    def prepare_splits(*, train_df, val_df, test_df):
        train_df["mol_dp"] = train_df.apply(get_molecule_datapoint, axis=1)
        val_df["mol_dp"] = val_df.apply(get_molecule_datapoint, axis=1)
        test_df["mol_dp"] = test_df.apply(get_molecule_datapoint, axis=1)

        featurizer = SimpleMoleculeMolGraphFeaturizer()
        train_mol_dataset = MoleculeDataset(
            train_df["mol_dp"].tolist(), featurizer=featurizer
        )
        val_mol_dataset = MoleculeDataset(
            val_df["mol_dp"].tolist(), featurizer=featurizer
        )
        test_mol_dataset = MoleculeDataset(
            test_df["mol_dp"].tolist(), featurizer=featurizer
        )

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
        model_config: BaselineConfig,
        X_d_scaler: StandardScaler | None,
        **kwargs,
    ) -> "ChempropRef":
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
        ffn = BinaryClassificationFFN(
            n_tasks=1,
            input_dim=ffn_dims,
            hidden_dim=model_config.ffn_hidden_dim,
            n_layers=model_config.ffn_n_layers,
            dropout=model_config.ffn_dropout,
        )  # type: ignore

        metric_list = [
            metrics.BinaryF1Score(),
            metrics.BinaryAUPRC(),
            metrics.BinaryAUROC(),
        ]

        model = MPNN(
            mp,
            agg,
            ffn,
            metrics=metric_list,
            X_d_transform=X_d_transform,
            batch_norm=model_config.batch_norm,
        )

        return ChempropRef(model)

    def train_func(
        self,
        *,
        train_split: MoleculeDataset,
        val_split: MoleculeDataset,
        train_config: TrainConfig,
        **kwargs,
    ) -> Self:
        set_seeds(train_config.random_seed)

        train_loader = build_dataloader(
            train_split,
            batch_size=train_config.batch_size,
            num_workers=8,
            seed=train_config.random_seed,
            worker_init_fn=seed_worker,
        )

        val_loader = build_dataloader(
            val_split, batch_size=train_config.batch_size, num_workers=8, shuffle=False
        )

        trainer = L.Trainer(
            logger=None,
            enable_checkpointing=True,
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            max_epochs=train_config.max_epochs,
            num_sanity_val_steps=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    verbose=True,
                    patience=train_config.early_stopping_patience,
                ),
                ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            ],
        )

        trainer.fit(
            self.model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        self.model = MPNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,  # type: ignore
            weights_only=False,
        )

        return self

    def tune_binary_classification_threshold(
        self,
        *,
        val_split: MoleculeDataset,
        val_labels,
        train_config: TrainConfig,
        **kwargs,
    ) -> float:
        pred_probs = get_pred_probs(self.model, val_split, scale_X_d=False)
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
        test_split: MoleculeDataset,
        binary_classification_threshold: float,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        pred_probs = get_pred_probs(self.model, test_split, scale_X_d=True)
        preds = (pred_probs >= binary_classification_threshold).astype(float)
        return pred_probs, preds
