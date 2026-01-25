import lightning as L
import numpy as np
import torch
from chemprop.data import MoleculeDataset
from chemprop.data.dataloader import collate_batch
from ghostml import optimize_threshold_from_predictions
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import move_data_to_device
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from sklearn.preprocessing import StandardScaler

from deltaprop.data import setup_train_val_dataloaders
from deltaprop.model import DeltaProp, build_model
from evaluate.data import set_seeds
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import interp1d

def train_func(
    config,
    train_mol_ds: MoleculeDataset,
    val_mol_ds: MoleculeDataset,
    binary_threshold: float,
    batch_size: int,
    random_seed: int,
    X_d_scaler: StandardScaler | None,
    max_epochs: int = 20,
    early_stopping_patience: int = 10,
):
    set_seeds(random_seed)
    train_dl, val_dl = setup_train_val_dataloaders(
        train_mol_ds, val_mol_ds, binary_threshold, batch_size, config["candidate_size"]
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

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    model = DeltaProp.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,  # type: ignore
        weights_only=False,
    )
    return model


search_space = {
    "depth": tune.qrandint(lower=2, upper=6, q=1),
    "ffn_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
    "ffn_num_layers": tune.qrandint(lower=1, upper=3, q=1),
    "message_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
    "encoder_dropout": tune.uniform(lower=0.0, upper=0.3),
    "interaction_dropout": tune.uniform(lower=0.0, upper=0.3),
    "batch_norm": tune.choice([True, False]),
    "candidate_size": 8,
}


def tune_func(
    config,
    train_mol_ds: MoleculeDataset,
    val_mol_ds: MoleculeDataset,
    binary_threshold: float,
    batch_size: int,
    random_seed: int,
    X_d_scaler: StandardScaler | None,
    max_epochs: int = 20,
    early_stopping_patience: int = 10,
):
    set_seeds(random_seed)
    train_dl, val_dl = setup_train_val_dataloaders(
        train_mol_ds, val_mol_ds, binary_threshold, batch_size, config["candidate_size"]
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



def get_prob(vals, tail_probs, binary_threshold):
    sort_idxs = np.argsort(vals)
    X = vals[sort_idxs]
    Y = tail_probs[sort_idxs]
    reg = IsotonicRegression(increasing=False, y_min=0, y_max=1.0)
    adj_Y = reg.fit_transform(X, Y)
    f = interp1d(X, adj_Y, kind="linear")
    return f(binary_threshold)


def get_interpolate_prob(pred_prob, exemplar_vals, binary_threshold):
    probs = []
    for idx in range(pred_prob.shape[0]):
        probs.append(get_prob(exemplar_vals, pred_prob[idx], binary_threshold))

    return np.array(probs)


def predict_func(
    model: DeltaProp,
    binary_classification_threshold: float,
    train_mol_ds: MoleculeDataset,
    test_mol_ds: MoleculeDataset,
):
    model.eval()

    train_embeds = embed_all(train_mol_ds, model)
    test_embeds = embed_all(test_mol_ds, model, scale_X_d=True)

    with torch.no_grad():
        pred_probs = (
            model.interaction(test_embeds, train_embeds)
            .sigmoid()
            .squeeze()
            .cpu()
            .numpy()
        )

    pos_mask = train_mol_ds.Y.squeeze() > 1.5
    neg_mask = ~pos_mask
    pos_contrib = pred_probs[:, pos_mask].mean(axis=-1)
    neg_contrib = pred_probs[:, neg_mask].mean(axis=-1)
    pred_probs = (pos_contrib + neg_contrib) / 2
    preds = (pred_probs >= binary_classification_threshold).astype(float)

    return pred_probs, preds


def tune_binary_classification_threshold(
    model: DeltaProp,
    train_mol_ds: MoleculeDataset,
    val_mol_ds: MoleculeDataset,
    labels,
    random_seed: int
):
    model.eval()

    train_embeds = embed_all(train_mol_ds, model)
    val_embeds = embed_all(val_mol_ds, model)

    with torch.no_grad():
        pred_probs = (
            model.interaction(val_embeds, train_embeds)
            .sigmoid()
            .squeeze()
            .cpu()
            .numpy()
        )

    pos_mask = train_mol_ds.Y.squeeze() > 1.5
    neg_mask = ~pos_mask
    pos_contrib = pred_probs[:, pos_mask].mean(axis=-1)
    neg_contrib = pred_probs[:, neg_mask].mean(axis=-1)
    pred_probs = (pos_contrib + neg_contrib) / 2

    thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
    optimal_threshold = optimize_threshold_from_predictions(
        labels=labels,
        probs=pred_probs,
        thresholds=thresholds,
        random_seed=random_seed,
    )

    return optimal_threshold
