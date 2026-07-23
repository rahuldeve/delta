import numpy as np
import torch
from chemprop.data import MoleculeDataset
from pytorch_lightning.utilities import move_data_to_device

from data import GT, DSThreshold


@torch.no_grad()
def embed_all(mol_dataset: MoleculeDataset, model, scale_X_d: bool = False):
    model.eval()
    if not scale_X_d:
        model.X_d_transform.train()

    # Imported lazily to avoid the data.py <-> utils.py circular import (data.py imports score_all).
    from models.deltaprop.data import delta_collate_batch

    dl = torch.utils.data.DataLoader(
        mol_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=delta_collate_batch,
        pin_memory=True,
        num_workers=4,
    )
    all_embeds = []
    for batch in dl:
        batch = move_data_to_device(batch, model.device)
        # field-name access keeps this agnostic to the extra bin_Y field
        all_embeds.append(model.encoding(batch.bmg, batch.V_d, batch.X_d).cpu())

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
    """Ranking-head strength score (projector output) per molecule, shape ``(N, 1)``."""
    embeds = embed_all(mol_dataset, model, scale_X_d)
    return model.interaction.projector(embeds.to(model.device)).cpu()


@torch.no_grad()
def classifier_probs(
    mol_dataset: MoleculeDataset, model, scale_X_d: bool = False
) -> "np.typing.NDArray[np.float64]":
    """Binary probabilities from the classification head, one per molecule.

    A direct ``sigmoid`` of the head output — no train reference set and no
    ``DSThreshold`` needed, since ``bin_y`` already encodes the positive-class
    direction. ``scale_X_d`` follows the same convention as :func:`embed_all`.
    """
    # Imported lazily for the same reason as in `embed_all`: model.py -> data.py ->
    # utils.py, so a module-level import of model.py here would be circular.
    from models.deltaprop.model import TiedClassifier

    embeds = embed_all(mol_dataset, model, scale_X_d).to(model.device)
    # The tied head reads the ranking strength λ; the independent head reads Z.
    head_input = (
        model.interaction.strength(embeds)
        if isinstance(model.classifier, TiedClassifier)
        else embeds
    )
    logits = model.classifier(head_input)
    return logits.sigmoid().cpu().numpy()


def build_discordancy_matrix(
    raw_scores: np.ndarray,
    reference_scores: np.ndarray,
    nu: float,
) -> np.ndarray:
    # Fail loudly if the model diverged — this is the only remaining NaN source
    # once the math below is stable, and silently swallowing it would hide it.
    if not np.isfinite(raw_scores).all():
        raise ValueError("raw_scores contains non-finite values (model diverged?)")

    # Net model preference for i over j, P(i≥j) − P(j≥i), in (-1, 1).
    # Derived from the Davidson form (exp(s_i) − exp(s_j)) /
    # (exp(s_i) + exp(s_j) + ν·exp((s_i+s_j)/2)) by dividing through by
    # exp((s_i+s_j)/2), which leaves a tanh/cosh expression in δ = s_i − s_j that
    # is finite for any score magnitude (no exp overflow).
    delta = raw_scores[:, None] - raw_scores[None, :]  # δ = s_i − s_j
    half = 0.5 * delta
    # cosh(half) may overflow to inf for very large |δ|; ν/inf → 0 is the intended
    # limit, so the result stays finite even though numpy warns on the overflow.
    net_model_pref = 2.0 * np.tanh(half) / (2.0 + nu / np.cosh(half))

    ref_diff = reference_scores[:, None] - reference_scores[None, :]

    D = -(ref_diff * net_model_pref)
    np.fill_diagonal(D, 0.0)
    return D


def top_k_discordant(
    D: np.ndarray,
    i: int,
    k: int,
    stochastic: bool = True,
    temperature: float = 2.0,
) -> list[int]:
    """
    Return up to k indices most discordant with index i.

    Args:
        D:           (N, N) discordancy matrix from build_discordancy_matrix.
        i:           Query index.
        k:           Maximum number of hard negatives to return.
        stochastic:  If True, sample proportional to discordancy (with temperature).
                     If False, return strict top-k by discordancy.
        temperature: Controls the sharpness of the sampling distribution.
                     - 1.0  → sample proportional to raw discordancy (default)
                     - >1.0 → flatter distribution, more exploration
                     - <1.0 → sharper distribution, closer to top-k behaviour
                     Must be > 0. Only used when stochastic=True.

    Returns:
        List of at most k indices j ≠ i, sorted by discordancy descending.
    """
    if not (0 <= i < D.shape[0]):
        raise IndexError(f"Index {i} out of bounds for matrix of size {D.shape[0]}.")
    if k <= 0:
        return []
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}.")

    row = D[i].copy()
    row[i] = 0.0
    row = np.maximum(row, 0.0)  # mask out concordant pairs

    total_discordancy = row.sum()
    if total_discordancy <= 0:
        return []

    if stochastic:
        row = row ** (1.0 / temperature)
        probs = row / row.sum()
        n_available = int((row > 0).sum())
        n_sample = min(k, n_available)
        chosen = np.random.choice(len(row), size=n_sample, replace=False, p=probs)
        chosen = sorted(chosen, key=lambda j: D[i, j], reverse=True)
    else:
        chosen = np.argsort(row)[::-1]
        chosen = [j for j in chosen if row[j] > 0][:k]

    return [int(j) for j in chosen]
