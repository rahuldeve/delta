import numpy as np
import torch
from chemprop.data import MoleculeDataset, collate_batch
from pytorch_lightning.utilities import move_data_to_device

from data import GT, DSThreshold


@torch.no_grad()
def project_lambdas(embeds: torch.Tensor, model) -> torch.Tensor:
    """Map embeddings to their scalar Davidson strengths λ = projector(embed).

    Returns a 1-D tensor on ``model.device``. ``squeeze(-1)`` (not a bare
    ``squeeze``) so a single-row input stays shape ``(1,)`` instead of collapsing
    to a scalar.
    """
    lam = model.interaction.projector(embeds.to(model.device)).squeeze(-1)
    return lam


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


def locate_lambda_tau(
    model,
    train_embeds: torch.Tensor,
    train_labels: "np.typing.NDArray[np.bool]",
    df_classification_threshold: DSThreshold,
) -> float:
    """Locate the boundary compound's strength λ_τ by the prevalence quantile.

    λ_τ is the strength the model would assign to a compound sitting exactly at the
    label cutoff τ. If a fraction ``π₊`` of train compounds are active and the model
    ranks faithfully, τ sits at the ``1 − π₊`` quantile of the train strengths for a
    GT task (top π₊ active) and the ``π₊`` quantile for LT (bottom π₊ active). ``π₊``
    comes from ``train_labels`` — the large split — so the location is stable even
    when positives are rare. Zero-parameter: only the class balance, no fit, so it
    can't suffer the flat-likelihood / separation degeneracy an MLE of the location
    would. See :func:`davidson_threshold_probs`.
    """
    lam_train = project_lambdas(train_embeds, model).float()
    y = np.asarray(train_labels).astype(bool)
    pi_pos = float(y.mean()) if y.size else 0.5
    pi_pos = min(max(pi_pos, 1e-6), 1.0 - 1e-6)  # guard all-one/all-zero folds
    q = 1.0 - pi_pos if isinstance(df_classification_threshold, GT) else pi_pos
    return float(torch.quantile(lam_train, q).item())


@torch.no_grad()
def davidson_threshold_probs(
    model,
    query_embeds: torch.Tensor,
    lam_tau: float,
    df_classification_threshold: DSThreshold,
) -> "np.typing.NDArray[np.float64]":
    """P(query active) as a single tie-centered Davidson comparison against λ_τ.

    ``P(active) = sigmoid(g(λ_q − λ_τ))`` for GT (active ≡ property ≥ τ ≡ the query
    beats the boundary), and the head/tail flip ``g(λ_τ − λ_q)`` for LT. ``g(δ) =
    δ + log(1 + ν·e^{−δ/2})`` is the Davidson logit; subtracting the tie baseline
    ``g(0) = log(1 + ν) > 0`` (a fixed offset, positive because ``>=`` counts a tie
    as a win) makes the score exactly 0 — hence ``P = 0.5`` — at ``λ_q == λ_τ``, so
    λ_τ *is* the 0.5 decision boundary by construction. Monotone in λ_q, so ROC-AUC /
    AP match the raw strength ordering; O(N_query), no reference set.
    """
    interaction = model.interaction
    lam_q = project_lambdas(query_embeds, model)  # (N_query,)
    lam_tau_t = torch.as_tensor(lam_tau, dtype=lam_q.dtype, device=lam_q.device)

    is_gt = isinstance(df_classification_threshold, GT)
    head = lam_q if is_gt else lam_tau_t
    tail = lam_tau_t if is_gt else lam_q
    g = interaction._davidson_logit(head, tail, interaction.log_nu)
    g0 = torch.logaddexp(torch.zeros_like(interaction.log_nu), interaction.log_nu)
    return (g - g0).sigmoid().cpu().numpy()


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
