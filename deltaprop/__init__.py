import lightning as L
import numpy as np
import torch
from chemprop.data import MoleculeDataset
from chemprop.data.dataloader import collate_batch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import move_data_to_device
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from deltaprop.data import setup_train_val_dataloaders
from deltaprop.model import DeltaProp, build_model
from utils import RANDOM_SEED, set_seeds


def train_func(
    config,
    train_mol_ds: MoleculeDataset,
    val_mol_ds: MoleculeDataset,
    binary_threshold: float,
    batch_size: int,
    X_d_scaler: StandardScaler | None,
    max_epochs: int = 20,
    early_stopping_patience: int = 10,
):
    set_seeds(RANDOM_SEED)
    train_dl, val_dl = setup_train_val_dataloaders(
        train_mol_ds, val_mol_ds, binary_threshold, batch_size, config["candidate_size"]
    )

    num_mol_feats = train_mol_ds.X_d.shape[-1] if train_mol_ds is not None else 0
    model = build_model(config, num_mol_feats, X_d_scaler)

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

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    model = DeltaProp.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # type: ignore
    return model


search_space = {
    "depth": tune.qrandint(lower=2, upper=6, q=1),
    "ffn_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
    "ffn_num_layers": tune.qrandint(lower=1, upper=3, q=1),
    "message_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
    "encoder_dropout": tune.uniform(lower=0.0, upper=0.3),
    "interaction_dropout": tune.uniform(lower=0.0, upper=0.3),
    "batch_norm": tune.choice([True, False]),
    "candidate_size": tune.qrandint(lower=4, upper=32, q=4),
}


def tune_func(
    config,
    train_mol_ds: MoleculeDataset,
    val_mol_ds: MoleculeDataset,
    binary_threshold: float,
    batch_size: int,
    X_d_scaler: StandardScaler | None,
    max_epochs: int = 20,
    early_stopping_patience: int = 10,
):
    set_seeds(RANDOM_SEED)
    train_dl, val_dl = setup_train_val_dataloaders(
        train_mol_ds, val_mol_ds, binary_threshold, batch_size, config["candidate_size"]
    )

    num_mol_feats = train_mol_ds.X_d.shape[-1] if train_mol_ds is not None else 0
    model = build_model(config, num_mol_feats, X_d_scaler)

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
            TuneReportCheckpointCallback(),
        ],
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


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


def evaluate_func(
    model: DeltaProp,
    train_mol_ds: MoleculeDataset,
    test_mol_ds: MoleculeDataset,
    binary_threshold: float,
):
    model.eval()

    train_embeds = embed_all(train_mol_ds, model)
    test_embeds = embed_all(test_mol_ds, model)

    exemplar_idxs = np.argwhere(train_mol_ds.Y.squeeze() > binary_threshold)
    exemplar_embeds = train_embeds[exemplar_idxs].squeeze()

    with torch.no_grad():
        pred_probs = (
            model.interaction(test_embeds, exemplar_embeds).sigmoid().mean(axis=-1)
        )
        preds = (pred_probs >= 0.5).float()

        pred_probs = pred_probs.cpu().detach().numpy().squeeze()
        preds = preds.cpu().detach().numpy().squeeze()
        labels = test_mol_ds.Y.squeeze() > binary_threshold

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "AUCROC",
        "PRAUC",
    ]
    metric_vals = [
        accuracy_score(labels, preds),
        balanced_accuracy_score(labels, preds),
        f1_score(labels, preds),
        precision_score(labels, preds),
        recall_score(labels, preds),
        roc_auc_score(labels, pred_probs),
        average_precision_score(labels, pred_probs),
    ]

    return {k: v for k, v in zip(metric_names, metric_vals)}
