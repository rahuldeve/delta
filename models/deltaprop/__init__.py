import random
from typing import Self

import lightning as L
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.dataloader import collate_batch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import (
    BondMessagePassing,
    NormAggregation,
    ScaleTransform,
)
from ghostml import optimize_threshold_from_predictions
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import move_data_to_device
from sklearn.preprocessing import StandardScaler

from config import TrainConfig
from misc import set_seeds
from models.abc import PreparedDatasetSplit, RefModel
from models.config import DeltapropConfig
from models.deltaprop.model import DeltaProp, Encoder, Interaction
from models.deltaprop.data import setup_train_val_dataloaders
from data import LT, DSThreshold
from scipy.special import expit


def get_molecule_datapoint(row):
    feat_entry_names = [f for f in row.index if f.startswith("feat")]
    if len(feat_entry_names) > 0:
        feat_array = pd.to_numeric(row[feat_entry_names], errors="coerce")
    else:
        feat_array = None

    return MoleculeDatapoint(
        mol=row["mol"], y=np.array([row["cont_target"]]), x_d=feat_array
    )



# ref: https://docs.pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def embed_all(mol_dataset: MoleculeDataset, model: DeltaProp, scale_X_d: bool = False):
    model.eval()
    if not scale_X_d:
        model.X_d_transform.train()

    dl = torch.utils.data.DataLoader(
        mol_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_batch,
    )
    all_embeds = []
    for batch in dl:
        batch = move_data_to_device(batch, model.device)
        res = model.embed_simple_batch(batch)
        all_embeds.append(res["embeds"])

    all_embeds = torch.cat(all_embeds)
    return all_embeds


class DeltapropRef(RefModel[DeltapropConfig]):
    def __init__(self, model: DeltaProp) -> None:
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
            activation='elu',
        )
        interaction = Interaction(encoder.output_dim, dropout=model_config.interaction_dropout)

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
            X_d_transform=X_d_transform,
            batch_norm=model_config.batch_norm,
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
        
        set_seeds(train_config.random_seed)
        train_dl, val_dl = setup_train_val_dataloaders(
            train_mol_ds=train_split, 
            val_mol_ds=val_split, 
            binary_threshold=df_classification_threshold, 
            batch_size=train_config.batch_size, 
            candidate_size=model_config.candidate_size
        )

        # from lightning.pytorch.loggers import WandbLogger
        # wandb_logger = WandbLogger(project="debug_gsk_hepg2", log_model="all", save_code=True)
        # wandb_logger.watch(self.model, log="gradients", log_freq=10) 
        # wandb_logger.experiment.mark_preempting()

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

        trainer.fit(self.model, train_dataloaders=train_dl, val_dataloaders=val_dl)
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

        train_embeds = embed_all(train_split, model)
        val_embeds = embed_all(val_split, model)

        with torch.no_grad():
            theta_hat_train = (
                model.interaction.projector(train_embeds)
                .squeeze()
                .cpu()
                .numpy()
            )

            theta_hat_val = (
                model.interaction.projector(val_embeds)
                .squeeze()
                .cpu()
                .numpy()
            )

        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        iso.fit(theta_hat_train, train_labels.squeeze())

        order = np.argsort(theta_hat_train)
        theta_k = np.interp(
            df_classification_threshold.th, 
            iso.predict(theta_hat_train[order]), 
            theta_hat_train[order]
        )

        pred_probs = expit(theta_hat_val - theta_k)

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
        **kwargs
    ):
        model = self.model
        model.eval()

        train_embeds = embed_all(train_split, model)
        test_embeds = embed_all(test_split, model, scale_X_d=True)

        with torch.no_grad():
            theta_hat_train = (
                model.interaction.projector(train_embeds)
                .squeeze()
                .cpu()
                .numpy()
            )

            theta_hat_test = (
                model.interaction.projector(test_embeds)
                .squeeze()
                .cpu()
                .numpy()
            )

        if isinstance(df_classification_threshold, LT):
            # by default, pred_probs[i, j] contains prob(i > j)
            # doing 1 - pred_probs will give us prob (i <= j)
            theta_hat_train = 1 - theta_hat_train
            theta_hat_test = 1 - theta_hat_test

        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        iso.fit(theta_hat_train, train_labels.squeeze())

        order = np.argsort(theta_hat_train)
        theta_k = np.interp(
            df_classification_threshold.th, 
            iso.predict(theta_hat_train[order]), 
            theta_hat_train[order]
        )

        pred_probs = expit(theta_hat_test - theta_k)
        preds = (pred_probs >= binary_classification_threshold).astype(float)

        return pred_probs, preds