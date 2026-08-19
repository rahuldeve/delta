import random
from itertools import chain
from typing import NamedTuple
import lightning as L
import numpy as np
import torch
from chemprop.data import MoleculeDataset, collate, dataloader, datasets
from torch.utils.data import DataLoader, Dataset

from data import GT, LT, DSThreshold
from models.deltaprop.utils import score_all


def build_hardness_matrix(
    raw_scores: np.ndarray,
    reference_scores: np.ndarray,
    nu: float,
    band: float = 1.0,
) -> np.ndarray:
    """Per-pair training value, peaking on pairs the model is undecided about.

    The signed agreement ``m = sign(Δy)·net_model_pref`` runs from -1 (model orders
    the pair confidently backwards) through 0 (undecided) to +1 (confidently right).
    Hardness is a tent centred on ``m = 0`` and reaching zero at ``|m| = band``, so
    sampling concentrates on pairs sitting *on* the decision boundary. ``band``
    is the half-width: 1.0 admits everything with a linear falloff, smaller values
    keep only the genuinely undecided.

    This deliberately down-weights both extremes. Confidently-right pairs carry no
    signal while Confidently-*wrong* pairs can be very noisy especially in initial 
    training phase. As training progresses, these confidently wrong pairs will get 
    surfaced.
    """
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

    # m in (-1, 1): how far the model agrees with the reference ordering.
    m = np.sign(ref_diff) * net_model_pref
    D = np.abs(ref_diff) * np.maximum(1.0 - np.abs(m) / band, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def top_k_hardest(
    D: np.ndarray,
    i: int,
    k: int,
    stochastic: bool = True,
    temperature: float = 1.0,
) -> list[int]:
    """
    Return up to k indices forming the hardest pairs with index i.

    Args:
        D:           (N, N) hardness matrix from build_hardness_matrix.
        i:           Query index.
        k:           Maximum number of hard negatives to return.
        stochastic:  If True, sample proportional to hardness (with temperature).
                     If False, return strict top-k by hardness.
        temperature: Controls the sharpness of the sampling distribution.
                     - 1.0  → sample proportional to raw hardness (default)
                     - >1.0 → flatter distribution, more exploration
                     - <1.0 → sharper distribution, closer to top-k behaviour
                     Must be > 0. Only used when stochastic=True.

    Returns:
        List of at most k indices j ≠ i, sorted by hardness descending.
    """
    if not (0 <= i < D.shape[0]):
        raise IndexError(f"Index {i} out of bounds for matrix of size {D.shape[0]}.")
    if k <= 0:
        return []
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}.")

    row = D[i].copy()
    row[i] = 0.0
    # build_hardness_matrix already clamps at 0, so this only guards against a
    # caller passing an unclamped matrix. Pairs outside the band sit at exactly 0
    # and are excluded by the `row > 0` count below.
    row = np.maximum(row, 0.0)

    total_hardness = row.sum()
    if total_hardness <= 0:
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


class RandomPairDataPoint(NamedTuple):
    anchor: datasets.Datum
    candidates: list[datasets.Datum]


class RandomPairTrainBatch(NamedTuple):
    anchor: collate.TrainingBatch
    candidates: collate.TrainingBatch
    B: int
    C: int


class RandomPairDataset(Dataset):
    def __init__(
        self,
        anchor_dataset: MoleculeDataset,
        candidate_dataset: MoleculeDataset,
        binary_threshold: DSThreshold,
        n_candidates: int,
        frac_hard: float = 0.2,
        hardness: np.ndarray | None = None,
    ):
        super().__init__()
        self.anchor_dataset = anchor_dataset
        self.candidate_dataset = candidate_dataset
        self.binary_threshold = binary_threshold
        self.n_candidates = n_candidates
        self.frac_hard = frac_hard
        self.hardness = hardness

        # Precompute the positive/negative candidate index pools once. They depend only on
        # candidate_dataset.Y and binary_threshold (both fixed for the dataset's lifetime), so
        # rebuilding them inside get_random_candidates on every __getitem__ was O(N) redundant
        # work per item -> O(N^2) per epoch in the dataloader workers. Computing them here also
        # lets forked workers inherit the arrays (copy-on-write) instead of each recomputing.
        self.pos_class_idxs, self.neg_class_idxs = self._class_index_pools()

    def __len__(self):
        return len(self.anchor_dataset)

    def get_hard_neg_idxs(self, idx: int, n: int) -> list[int]:
        if self.hardness is None:
            return []

        if n == 0:
            return []

        return top_k_hardest(self.hardness, idx, n)

    def _class_index_pools(self) -> tuple[np.ndarray, np.ndarray]:
        """Indices of candidate molecules in the positive / negative class.

        Depends only on candidate_dataset.Y and binary_threshold, both fixed at construction,
        so this is computed once (see __init__) rather than on every __getitem__.
        """
        targets = self.candidate_dataset.Y.squeeze()
        if isinstance(self.binary_threshold, GT):
            pos_class_mask = targets >= self.binary_threshold.th
        elif isinstance(self.binary_threshold, LT):
            pos_class_mask = targets <= self.binary_threshold.th
        else:
            raise ValueError(self.binary_threshold)

        pos_class_idxs = np.argwhere(pos_class_mask).squeeze()
        neg_class_idxs = np.argwhere(~pos_class_mask).squeeze()
        return pos_class_idxs, neg_class_idxs

    def get_random_idxs(
        self, n_random: int, exclude: set[int] | None = None
    ) -> list[int]:
        pos_class_idxs = self.pos_class_idxs
        neg_class_idxs = self.neg_class_idxs

        # Drop any indices already chosen as hard negatives so they are not drawn
        # again here, which would put duplicate candidates in the same anchor's set.
        if exclude:
            exclude_arr = np.fromiter(exclude, dtype=pos_class_idxs.dtype)
            pos_class_idxs = pos_class_idxs[~np.isin(pos_class_idxs, exclude_arr)]
            neg_class_idxs = neg_class_idxs[~np.isin(neg_class_idxs, exclude_arr)]

        pos_class_sample_count = min(int(0.5 * n_random), pos_class_idxs.shape[0])
        random_pos_class_idxs = np.random.choice(
            pos_class_idxs,
            size=(pos_class_sample_count,),
            replace=False,
        )

        neg_class_sample_count = min(
            n_random - pos_class_sample_count, neg_class_idxs.shape[0]
        )
        random_neg_class_idxs = np.random.choice(
            neg_class_idxs,
            size=(neg_class_sample_count,),
            replace=False,
        )

        return [int(idx) for idx in chain(random_pos_class_idxs, random_neg_class_idxs)]

    def __getitem__(self, idx) -> RandomPairDataPoint:
        hard_neg_idxs = self.get_hard_neg_idxs(
            idx, int(self.frac_hard * self.n_candidates)
        )
        random_idxs = self.get_random_idxs(
            self.n_candidates - len(hard_neg_idxs),
            exclude=set(hard_neg_idxs),
        )
        candidates = [
            self.candidate_dataset[j] for j in chain(hard_neg_idxs, random_idxs)
        ]
        return RandomPairDataPoint(
            self.anchor_dataset[idx],
            candidates,
        )

    @staticmethod
    def collate_function(batch):
        batch_anchors, batch_exemplars = zip(*batch)
        B = len(batch)
        C = len(batch_exemplars[0])
        batch_anchors = dataloader.collate_batch(batch_anchors)
        batch_exemplars = dataloader.collate_batch(chain.from_iterable(batch_exemplars))
        return RandomPairTrainBatch(batch_anchors, batch_exemplars, B, C)


class RandomPairDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_mol_ds: MoleculeDataset,
        val_mol_ds: MoleculeDataset,
        binary_threshold: DSThreshold,
        batch_size: int,
        n_candidates: int,
        frac_hard: float = 0.2,
        hard_band: float = 1.0,
        num_workers: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.train_ds = RandomPairDataset(
            anchor_dataset=train_mol_ds,
            candidate_dataset=train_mol_ds,
            binary_threshold=binary_threshold,
            n_candidates=n_candidates,
            frac_hard=frac_hard,
        )

        self.val_ds = RandomPairDataset(
            anchor_dataset=val_mol_ds,
            candidate_dataset=train_mol_ds,
            binary_threshold=binary_threshold,
            n_candidates=n_candidates,
            frac_hard=frac_hard,
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hard_band = hard_band
        self.seed = seed

    def train_dataloader(self):
        # Rebuilt here, so this must stay in step with the trainer's
        # reload_dataloaders_every_n_epochs: at n=2 the matrix only refreshed on
        # even epochs and odd epochs re-trained on a pool mined against a
        # two-epoch-old model. Keep that setting at 1.
        if self.trainer is not None and self.trainer.current_epoch > 0:
            # assert self.trainer.model is not None
            train_mol_ds = self.train_ds.anchor_dataset
            model = self.trainer.model
            theta_hat_train = score_all(train_mol_ds, model).squeeze()

            with torch.no_grad():
                nu = model.interaction.log_nu.exp().cpu().item()  # type: ignore

            self.update_hardness_mat_train(
                theta_hat_train.cpu().numpy(),
                nu,
            )

        # Seed the generator per-epoch so each epoch draws a distinct (but
        # reproducible) RNG stream. With a constant seed the worker seeds — and
        # thus the shuffle order and random-candidate draws — would repeat every
        # reload (reload_dataloaders_every_n_epochs), collapsing the resampled
        # pairs to a couple of fixed realizations across the whole run.
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=RandomPairDataset.collate_function,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed + epoch),
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self):
        # Seed the workers so the (random) validation candidate pairs are reproducible
        # across runs
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=RandomPairDataset.collate_function,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
        )

    def update_hardness_mat_train(self, train_model_scores, nu: float):
        reference_scores = self.train_ds.anchor_dataset.Y.squeeze()
        self.train_ds.hardness = build_hardness_matrix(
            train_model_scores, reference_scores, nu, self.hard_band
        )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
