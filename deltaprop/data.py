import random
from itertools import chain
from typing import NamedTuple

import lightning as L
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
    def __init__(
        self,
        anchor_dataset: data.datasets.MoleculeDataset,
        candidate_dataset: data.datasets.MoleculeDataset,
        binary_threshold: float,
        n_candidates: int,
        hard_neg_idxs: list[list[int]] | None=None,
    ):
        super().__init__()
        self.anchor_dataset = anchor_dataset
        self.candidate_dataset = candidate_dataset
        self.n_candidates = n_candidates
        self.binary_threshold = binary_threshold
        self.hard_neg_idxs = hard_neg_idxs

    def __len__(self):
        return len(self.anchor_dataset)

    def get_hard_neg_candidate_idxs(self, idx: int):
        if self.hard_neg_idxs is None:
            return []

        hard_neg_candidate_idxs = self.hard_neg_idxs[idx]

        selected_idxs = np.random.choice(
            hard_neg_candidate_idxs,
            size=(min(self.n_candidates, len(hard_neg_candidate_idxs)), ),
            replace=False,
        )

        return selected_idxs.tolist()

    def get_random_candidate_idxs(self, selected_hard_neg_idxs):
        remaining_idxs = (
            set(range(len(self.candidate_dataset))) - 
            set(selected_hard_neg_idxs)
        )

        n = 2*self.n_candidates - len(selected_hard_neg_idxs)
        selected_idxs = np.random.choice(list(remaining_idxs), size=(n,), replace=False)
        return selected_idxs.tolist()

    def __getitem__(self, idx) -> RandomPairDataPoint:
        hard_neg_idxs = self.get_hard_neg_candidate_idxs(idx)
        random_idxs = self.get_random_candidate_idxs(hard_neg_idxs)

        return RandomPairDataPoint(
            self.anchor_dataset[idx],
            [self.candidate_dataset[idx] for idx in hard_neg_idxs + random_idxs]
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


class RandomPairDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_mol_ds: MoleculeDataset,
        val_mol_ds: MoleculeDataset,
        binary_threshold: float,
        batch_size: int,
        candidate_size: int,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.train_ds = RandomPairDataset(
            train_mol_ds, train_mol_ds, binary_threshold, candidate_size
        )

        self.val_ds = RandomPairDataset(
            train_mol_ds, val_mol_ds, binary_threshold, candidate_size
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=RandomPairDataset.collate_function,
            worker_init_fn=seed_worker,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=RandomPairDataset.collate_function,
            num_workers=self.num_workers,
        )
    
    


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
    train_pair_ds = RandomPairDataset(
        train_mol_ds, train_mol_ds, binary_threshold, candidate_size
    )
    val_pair_ds = RandomPairDataset(
        train_mol_ds, val_mol_ds, binary_threshold, candidate_size
    )

    train_pair_dl = DataLoader(
        train_pair_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=RandomPairDataset.collate_function,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        drop_last=True,
    )

    val_pair_dl = DataLoader(
        val_pair_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=RandomPairDataset.collate_function,
        num_workers=num_workers,
    )

    return train_pair_dl, val_pair_dl
