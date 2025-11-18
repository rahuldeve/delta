import random
from itertools import chain
from typing import NamedTuple

import numpy as np
import torch
from chemprop import data
from chemprop.data import MoleculeDataset
from torch.utils.data import DataLoader, Dataset


class RandomPairDataPoint(NamedTuple):
    anchor: data.datasets.Datum
    candidates: list[data.datasets.Datum]


class RandomPairTrainBatch(NamedTuple):
    anchor: data.collate.TrainingBatch
    candidates: data.collate.TrainingBatch
    B: int
    C: int


class RandomPairDataset(Dataset):
    def __init__(self, anchor_dataset, candidate_dataset, binary_threshold, n_candidates):
        super().__init__()
        self.anchor_dataset: data.datasets.MoleculeDataset = anchor_dataset
        self.candidate_dataset: data.datasets.MoleculeDataset = candidate_dataset
        self.n_candidates: int = n_candidates
        self.binary_threshold = binary_threshold

    def __len__(self):
        return len(self.anchor_dataset)

    def get_exemplar_candidates(self):
        targets = self.candidate_dataset.Y.squeeze()
        exemplar_mask = targets > self.binary_threshold
        exemplar_idxs = np.argwhere(exemplar_mask).squeeze()

        selected_idxs = np.random.choice(
            exemplar_idxs,
            size=(min(self.n_candidates, exemplar_idxs.shape[0]), ),
            replace=False
        )

        return [self.candidate_dataset[idx] for idx in selected_idxs]

    def get_random_candidates(self, n):
        targets = self.candidate_dataset.Y.squeeze()
        exemplar_mask = targets > self.binary_threshold
        non_exemplar_idxs = np.argwhere(~exemplar_mask).squeeze()

        candidate_idxs = np.random.choice(
            non_exemplar_idxs, size=(n,), replace=False
        )
        return [self.candidate_dataset[idx] for idx in candidate_idxs]

    def __getitem__(self, idx) -> RandomPairDataPoint:
        exemplar_idxs = self.get_exemplar_candidates()
        random_idxs = self.get_random_candidates(2*self.n_candidates - len(exemplar_idxs))
        return RandomPairDataPoint(
            self.anchor_dataset[idx],
            exemplar_idxs + random_idxs,
        )

    @staticmethod
    def collate_function(batch):
        batch_anchors, batch_exemplars = zip(*batch)
        B = len(batch)
        C = len(batch_exemplars[0])
        batch_anchors = data.dataloader.collate_batch(batch_anchors)
        batch_exemplars = data.dataloader.collate_batch(
            chain.from_iterable(batch_exemplars)
        )
        return RandomPairTrainBatch(batch_anchors, batch_exemplars, B, C)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_train_val_dataloaders(
    train_mol_ds: MoleculeDataset,
    val_mol_ds: MoleculeDataset,
    binary_threshold: float,
    batch_size: int,
    candidate_size: int,
    num_workers: int = 8,
):
    train_pair_ds = RandomPairDataset(train_mol_ds, train_mol_ds, binary_threshold, candidate_size)
    val_pair_ds = RandomPairDataset(train_mol_ds, val_mol_ds, binary_threshold, candidate_size)

    train_pair_dl = DataLoader(
        train_pair_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=RandomPairDataset.collate_function,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        drop_last=True
    )

    val_pair_dl = DataLoader(
        val_pair_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=RandomPairDataset.collate_function,
        num_workers=num_workers,
    )

    return train_pair_dl, val_pair_dl
