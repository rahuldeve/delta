"""Equivalence test for the chunked inference helper.

`class_mean_probs` streams the query molecules in chunks so the GPU never holds
the full ``(N_query x N_train)`` interaction matrix. This checks it produces the
same per-query class-averaged probabilities as the original single-shot path that
materialized the whole matrix.

No model is trained: we use a real (randomly initialized) ``Interaction`` module
together with random embeddings and random binary labels, which exercises the
projector, ``_davidson_logit``, the GT/LT argument flip, and the pos/neg
three-way reduction exactly as production does.

Run standalone:  uv run --active python tests/test_class_mean_probs.py
(or under pytest, if installed)
"""

import sys
from pathlib import Path

import numpy as np
import torch

# allow running as a standalone script: put the repo root on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import GT, LT
from models.deltaprop.model import Interaction
from models.deltaprop.utils import class_mean_probs


class _FakeModel:
    """Minimal stand-in exposing the only attributes the helper touches."""

    def __init__(self, interaction: Interaction, device: torch.device):
        self.interaction = interaction
        self.device = device


def _reference(model, train_embeds, query_embeds, train_labels, dsth):
    """Original full-matrix implementation copied from the pre-refactor methods."""
    interaction = model.interaction
    with torch.no_grad():
        theta_train = interaction.projector(train_embeds).squeeze().unsqueeze(0)
        theta_query = interaction.projector(query_embeds).squeeze().unsqueeze(1)

        if isinstance(dsth, GT):
            pred = (
                interaction._davidson_logit(
                    theta_query, theta_train, interaction.log_nu
                )
                .sigmoid()
                .squeeze()
                .cpu()
                .numpy()
            )
        else:
            pred = (
                interaction._davidson_logit(
                    theta_train, theta_query, interaction.log_nu
                )
                .sigmoid()
                .squeeze()
                .cpu()
                .numpy()
            )

    pos_mask = train_labels
    neg_mask = ~pos_mask
    pos_contrib = pred[:, pos_mask]
    neg_contrib = pred[:, neg_mask]
    if pos_contrib.shape[-1] == 0:
        return neg_contrib.mean(axis=-1)
    elif neg_contrib.shape[-1] == 0:
        return pos_contrib.mean(axis=-1)
    else:
        return (pred[:, pos_mask].mean(axis=-1) + pred[:, neg_mask].mean(axis=-1)) / 2


def _make_case(seed, n_train, n_query, ndims, label_kind):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    interaction = Interaction(ndims).eval()
    model = _FakeModel(interaction, torch.device("cpu"))

    train_embeds = torch.randn(n_train, ndims)
    query_embeds = torch.randn(n_query, ndims)

    if label_kind == "mixed":
        labels = rng.integers(0, 2, size=n_train).astype(bool)
        # guarantee both classes are present
        labels[0] = True
        labels[1] = False
    elif label_kind == "all_pos":
        labels = np.ones(n_train, dtype=bool)
    elif label_kind == "all_neg":
        labels = np.zeros(n_train, dtype=bool)
    else:
        raise ValueError(label_kind)

    return model, train_embeds, query_embeds, labels


def run_all():
    cases = []
    seed = 0
    for dsth in (GT(th=0.0), LT(th=0.0)):
        for label_kind in ("mixed", "all_pos", "all_neg"):
            # chunk sizes spanning: tiny, non-divisor, exactly N_query, larger than N_query
            for chunk_size in (1, 7, 50, 1024):
                seed += 1
                model, train_embeds, query_embeds, labels = _make_case(
                    seed=seed, n_train=37, n_query=50, ndims=8, label_kind=label_kind
                )

                ref = _reference(model, train_embeds, query_embeds, labels, dsth)
                got = class_mean_probs(
                    model,
                    train_embeds,
                    query_embeds,
                    labels,
                    dsth,
                    chunk_size=chunk_size,
                )

                assert got.shape == ref.shape, (
                    f"shape mismatch {got.shape} vs {ref.shape} "
                    f"(dsth={type(dsth).__name__}, labels={label_kind}, chunk={chunk_size})"
                )
                assert np.allclose(got, ref, atol=1e-6), (
                    f"value mismatch (max abs diff "
                    f"{np.abs(got - ref).max():.2e}) for "
                    f"dsth={type(dsth).__name__}, labels={label_kind}, chunk={chunk_size}"
                )
                cases.append((type(dsth).__name__, label_kind, chunk_size))

    return cases


def test_class_mean_probs_matches_full_matrix():
    """Pytest entry point."""
    assert run_all()


if __name__ == "__main__":
    passed = run_all()
    print(f"OK: {len(passed)} cases matched the full-matrix reference")
