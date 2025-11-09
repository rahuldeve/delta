import random

import lightning as L
import numpy as np
import torch
from chemprop.data import MoleculeDataset, build_dataloader
from chemprop.data.dataloader import collate_batch
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
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from sklearn.preprocessing import StandardScaler

from utils import RANDOM_SEED, set_seeds


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(config, X_d_scaler: StandardScaler | None):
    depth = config["depth"]
    ffn_hidden_dim = config["ffn_hidden_dim"]
    ffn_num_layers = config["ffn_num_layers"]
    message_hidden_dim = config["message_hidden_dim"]
    batch_norm = config["batch_norm"]

    if X_d_scaler is not None:
        X_d_transform = ScaleTransform.from_standard_scaler(X_d_scaler)
        num_mol_feats = X_d_scaler.n_features_in_
    else:
        X_d_transform = None
        num_mol_feats = 0

    mp = BondMessagePassing(d_h=message_hidden_dim, depth=depth)  # type: ignore
    agg = NormAggregation()
    ffn_dims = mp.output_dim + num_mol_feats
    ffn = BinaryClassificationFFN(
        n_tasks=1,
        input_dim=ffn_dims,
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_num_layers,
    )  # type: ignore
    metric_list = [
        metrics.BinaryF1Score(),
        metrics.BinaryAUPRC(),
        metrics.BinaryAUROC(),
    ]
    return MPNN(mp, agg, ffn, batch_norm, metric_list, X_d_transform=X_d_transform)


def train_func(
    config,
    train_mol_ds: MoleculeDataset,
    val_mol_ds: MoleculeDataset,
    batch_size: int,
    X_d_scaler: StandardScaler | None,
    max_epochs: int = 20,
    early_stopping_patience: int = 10,
):
    set_seeds(RANDOM_SEED)

    train_loader = build_dataloader(
        train_mol_ds,
        batch_size=batch_size,
        num_workers=8,
        seed=RANDOM_SEED,
        worker_init_fn=seed_worker,
    )

    val_loader = build_dataloader(
        val_mol_ds, batch_size=batch_size, num_workers=8, shuffle=False
    )

    model = build_model(config, X_d_scaler)

    trainer = L.Trainer(
        logger=None,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=True,
                patience=early_stopping_patience,
            ),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        ],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = MPNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # type: ignore
    return model


search_space = {
    "depth": tune.qrandint(lower=2, upper=6, q=1),
    "ffn_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
    "ffn_num_layers": tune.qrandint(lower=1, upper=3, q=1),
    "message_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
    "batch_norm": tune.choice([True, False]),
}


def tune_func(
    config,
    train_mol_ds: MoleculeDataset,
    val_mol_ds: MoleculeDataset,
    batch_size: int,
    X_d_scaler: StandardScaler | None,
    max_epochs: int = 20,
    early_stopping_patience: int = 10,
):
    set_seeds(RANDOM_SEED)

    train_loader = build_dataloader(
        train_mol_ds,
        batch_size=batch_size,
        num_workers=8,
        seed=RANDOM_SEED,
        worker_init_fn=seed_worker,
    )

    val_loader = build_dataloader(
        val_mol_ds, batch_size=batch_size, num_workers=8, shuffle=False
    )

    model = build_model(config, X_d_scaler)

    trainer = L.Trainer(
        logger=None,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=False,
                patience=early_stopping_patience,
            ),
            TuneReportCheckpointCallback(save_checkpoints=False),
        ],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


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


def predict_func(
    model: MPNN,
    binary_classification_threshold: float,
    test_mol_ds: MoleculeDataset,
):
    pred_probs = get_pred_probs(model, test_mol_ds, scale_X_d=True)
    preds = (pred_probs >= binary_classification_threshold).astype(float)
    return pred_probs, preds


def tune_binary_classification_threshold(
    model: MPNN, val_mol_ds: MoleculeDataset, val_labels
):
    pred_probs = get_pred_probs(model, val_mol_ds, scale_X_d=False)
    thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)

    optimal_threshold = optimize_threshold_from_predictions(
        labels=val_labels,
        probs=pred_probs,
        thresholds=thresholds,
        random_seed=RANDOM_SEED,
    )

    return optimal_threshold
