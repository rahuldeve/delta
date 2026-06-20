import numpy as np
import torch
from chemprop.data import MoleculeDataset, collate_batch
from pytorch_lightning.utilities import move_data_to_device

from data import GT, DSThreshold


@torch.no_grad()
def embed_all(mol_dataset: MoleculeDataset, model, scale_X_d: bool = False):
    model.eval()
    if not scale_X_d:
        model.X_d_transform.train()

    dl = torch.utils.data.DataLoader(
        mol_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=True,
        num_workers=4,
    )
    all_embeds = []
    for batch in dl:
        batch = move_data_to_device(batch, model.device)
        res = model.embed_simple_batch(batch)
        all_embeds.append(res["embeds"].cpu())

    all_embeds = torch.cat(all_embeds)
    return all_embeds


@torch.no_grad()
def class_mean_probs(
    model,
    train_embeds: torch.Tensor,
    query_embeds: torch.Tensor,
    train_labels: "np.typing.NDArray[np.bool]",
    df_classification_threshold: DSThreshold,
    chunk_size: int = 1024,
) -> "np.typing.NDArray[np.float64]":
    """Per-query class-averaged interaction probability against the train set.

    Streams the query molecules in chunks so the GPU only ever holds a
    ``(chunk_size, N_train)`` interaction tensor instead of the full
    ``(N_query, N_train)`` matrix. Numerically equivalent to the single-shot path.
    """
    device = model.device
    interaction = model.interaction

    theta_train = interaction.projector(train_embeds.to(device)).squeeze()  # (N_train,)
    theta_query = interaction.projector(query_embeds.to(device)).squeeze()  # (N_query,)

    pos_mask = torch.as_tensor(train_labels, dtype=torch.bool, device=device)
    neg_mask = ~pos_mask
    has_pos = bool(pos_mask.any())
    has_neg = bool(neg_mask.any())
    is_gt = isinstance(df_classification_threshold, GT)

    results = []
    for start in range(0, theta_query.shape[0], chunk_size):
        theta_q = theta_query[start : start + chunk_size]  # (chunk,)

        head = theta_q[:, None] if is_gt else theta_train[None, :]
        tail = theta_train[None, :] if is_gt else theta_q[:, None]
        probs = interaction._davidson_logit(
            head, tail, interaction.log_nu
        ).sigmoid()  # (chunk, N_train)

        if has_pos and has_neg:
            reduced = (
                probs[:, pos_mask].mean(dim=-1) + probs[:, neg_mask].mean(dim=-1)
            ) / 2
        elif has_pos:
            reduced = probs[:, pos_mask].mean(dim=-1)
        else:
            reduced = probs[:, neg_mask].mean(dim=-1)

        results.append(reduced.cpu().numpy())

    return np.concatenate(results)


@torch.no_grad()
def score_all(mol_dataset: MoleculeDataset, model, scale_X_d: bool = False):
    model.eval()
    if not scale_X_d:
        model.X_d_transform.train()

    dl = torch.utils.data.DataLoader(
        mol_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=True,
        num_workers=4,
    )
    all_scores = []
    for batch in dl:
        batch = move_data_to_device(batch, model.device)
        res = model.embed_simple_batch(batch)
        scores = model.interaction.projector(res["embeds"])
        all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores)
    return all_scores
