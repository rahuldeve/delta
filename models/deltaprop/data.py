import random
from dataclasses import dataclass
from itertools import chain
from typing import NamedTuple

import lightning as L
import numpy as np
import torch
from chemprop.data import (
    BatchMolGraph,
    MoleculeDatapoint,
    MolGraph,
    MoleculeDataset,
    dataloader
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data import GT, LT, DSThreshold
from models.deltaprop.utils import build_discordancy_matrix, score_all, top_k_discordant


class DeltaDatum(NamedTuple):
    """A single training data point carrying both a continuous and a binary target.

    Mirrors :class:`chemprop.data.Datum` with its single ``y`` replaced by
    ``reg_y`` (continuous / regression target) and ``bin_y`` (binary label).
    """

    mg: MolGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    reg_y: np.ndarray | None
    bin_y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


@dataclass
class DeltaMoleculeDatapoint(MoleculeDatapoint):
    """A :class:`~chemprop.data.MoleculeDatapoint` that also carries a binary target.

    The inherited ``y`` holds the continuous target — by default ``Y`` (and hence
    :attr:`DeltaDatum.reg_y`) is pegged to the regression target — so chemprop's
    ``_Y``/``Y``/``normalize_targets`` machinery keeps working unchanged. The binary
    label rides alongside it in ``bin_y``.
    """

    bin_y: np.ndarray | None = None


class DeltaMoleculeDataset(MoleculeDataset):
    """A :class:`~chemprop.data.MoleculeDataset` that yields :class:`DeltaDatum`\\s.

    Adds a parallel ``bin_Y`` binary-target array (backed by each datapoint's
    ``bin_y``) next to the inherited continuous ``Y``, following the same pattern
    chemprop uses for the extra target arrays in ``MolAtomBondDataset``. By default
    ``Y`` is pegged to the regression target.
    """

    data: list[DeltaMoleculeDatapoint]  # type: ignore

    def __getitem__(self, idx: int) -> DeltaDatum:  # type: ignore
        d = self.data[idx]
        mg = self.mg_cache[idx]

        return DeltaDatum(
            mg,
            self.V_ds[idx],
            self.X_d[idx],
            # by default Y is pegged to the regression target (the datapoint's `y`)
            self.Y[idx],
            self.bin_Y[idx],
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    @property
    def _bin_Y(self) -> np.ndarray:
        """the raw binary targets of the dataset"""
        return np.array([d.bin_y for d in self.data], float)

    @property
    def bin_Y(self) -> np.ndarray:
        """the binary targets of the dataset"""
        return self.__bin_Y

    @bin_Y.setter
    def bin_Y(self, bin_Y):
        self._validate_attribute(bin_Y, "binary targets")

        self.__bin_Y = np.array(bin_Y, float)

    def reset(self):
        """Reset the dataset's targets and features to their initial values.

        ``MoleculeDataset.__post_init__`` calls ``reset()``, so this is what
        populates ``bin_Y`` at construction time.
        """
        super().reset()
        self.__bin_Y = self._bin_Y


class DeltaTrainingBatch(NamedTuple):
    """A batch of :class:`DeltaDatum`\\s.

    Mirrors :class:`chemprop.data.TrainingBatch` with its single ``Y`` replaced by
    ``reg_Y`` (continuous) and ``bin_Y`` (binary).
    """

    bmg: BatchMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    reg_Y: Tensor | None
    bin_Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def delta_collate_batch(batch) -> DeltaTrainingBatch:
    """Collate an iterable of :class:`DeltaDatum`\\s into a :class:`DeltaTrainingBatch`.

    Mirrors :func:`chemprop.data.collate_batch`; the unpack order matches
    :class:`DeltaDatum`.
    """
    mgs, V_ds, x_ds, reg_ys, bin_ys, weights, lt_masks, gt_masks = zip(*batch)

    return DeltaTrainingBatch(
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if reg_ys[0] is None else torch.from_numpy(np.array(reg_ys)).float(),
        None if bin_ys[0] is None else torch.from_numpy(np.array(bin_ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


class RandomPairDataPoint(NamedTuple):
    anchor: DeltaDatum
    candidates: list[DeltaDatum]


class RandomPairTrainBatch(NamedTuple):
    anchor: DeltaTrainingBatch
    candidates: DeltaTrainingBatch
    B: int
    C: int


class RandomPairDataset(Dataset):
    def __init__(
        self,
        anchor_dataset: DeltaMoleculeDataset,
        candidate_dataset: DeltaMoleculeDataset,
        binary_threshold: DSThreshold,
        n_candidates: int,
        frac_hard: float = 0.2,
        discordancy_degree: np.ndarray | None = None,
    ):
        super().__init__()
        self.anchor_dataset = anchor_dataset
        self.candidate_dataset = candidate_dataset
        self.binary_threshold = binary_threshold
        self.n_candidates = n_candidates
        self.frac_hard = frac_hard
        self.discordancy_degree = discordancy_degree

        # Precompute the positive/negative candidate index pools once. They depend only on
        # candidate_dataset.Y and binary_threshold (both fixed for the dataset's lifetime), so
        # rebuilding them inside get_random_candidates on every __getitem__ was O(N) redundant
        # work per item -> O(N^2) per epoch in the dataloader workers. Computing them here also
        # lets forked workers inherit the arrays (copy-on-write) instead of each recomputing.
        self.pos_class_idxs, self.neg_class_idxs = self._class_index_pools()

    def __len__(self):
        return len(self.anchor_dataset)

    def get_hard_neg_candidates(self, idx: int, n: int):
        if self.discordancy_degree is None:
            return []

        if n == 0:
            return []

        selected_idxs = top_k_discordant(self.discordancy_degree, idx, n)

        return [self.candidate_dataset[idx] for idx in selected_idxs]

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

    def get_random_candidates(self, n_random: int):
        pos_class_idxs = self.pos_class_idxs
        neg_class_idxs = self.neg_class_idxs

        pos_class_sample_count = min(int(0.8 * n_random), pos_class_idxs.shape[0])
        random_pos_class_idxs = np.random.choice(
            pos_class_idxs,
            size=(pos_class_sample_count,),
            replace=False,
        )
        random_pos_candidates = [
            self.candidate_dataset[idx] for idx in random_pos_class_idxs
        ]

        neg_class_sample_count = min(
            n_random - pos_class_sample_count, neg_class_idxs.shape[0]
        )
        random_neg_class_idxs = np.random.choice(
            neg_class_idxs,
            size=(neg_class_sample_count,),
            replace=False,
        )
        random_neg_candidates = [
            self.candidate_dataset[idx] for idx in random_neg_class_idxs
        ]

        return random_pos_candidates + random_neg_candidates

    def __getitem__(self, idx) -> RandomPairDataPoint:
        hard_neg_candidates = self.get_hard_neg_candidates(
            idx, int(self.frac_hard * self.n_candidates)
        )
        random_candidates = self.get_random_candidates(
            self.n_candidates - len(hard_neg_candidates)
        )
        # random_candidates = self.get_random_candidates(self.n_candidates)
        return RandomPairDataPoint(
            self.anchor_dataset[idx],
            hard_neg_candidates + random_candidates,
            # random_candidates
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
        self.seed = seed

    def train_dataloader(self):
        if self.trainer is not None and self.trainer.current_epoch > 0:
            # assert self.trainer.model is not None
            train_mol_ds = self.train_ds.anchor_dataset
            model = self.trainer.model
            theta_hat_train = score_all(train_mol_ds, model).squeeze()

            with torch.no_grad():
                nu = model.interaction.log_nu.exp().cpu().item()  # type: ignore

            self.update_discordancy_mat_train(
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
        # Seed the workers so the (random) validation candidate pairs are
        # reproducible across runs, making val_loss a stable early-stopping signal.
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=RandomPairDataset.collate_function,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
        )

    def update_discordancy_mat_train(self, train_model_scores, nu: float):
        reference_scores = self.train_ds.anchor_dataset.Y.squeeze()
        self.train_ds.discordancy_degree = build_discordancy_matrix(
            train_model_scores, reference_scores, nu
        )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
