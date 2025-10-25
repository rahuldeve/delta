from typing import NamedTuple
import numpy as np
from chemprop import data
from torch.utils.data import Dataset, DataLoader
import torch
from itertools import chain
import random
import lightning as L


class RandomPairDataPoint(NamedTuple):
    anchor: data.datasets.Datum
    exemplar: list[data.datasets.Datum]


class RandomPairTrainBatch(NamedTuple):
    anchor: data.collate.TrainingBatch
    exemplar: data.collate.TrainingBatch
    B: int
    C: int


class RandomPairDataset(Dataset):
    def __init__(self, mol_dataset, n_candidates):
        super().__init__()
        self.mol_dataset: data.datasets.MoleculeDataset = mol_dataset
        self.n_candidates: int = n_candidates

    def __len__(self):
        return len(self.mol_dataset)

    def get_exemplar_candidates(self):
        targets = self.mol_dataset.Y.squeeze()
        mask = targets > 50
        weights = np.where(mask, 1.0, 0.0)
        probs = weights / weights.sum()
        exemplar_idxs = np.random.choice(
            targets.shape[0], size=(self.n_candidates,), p=probs, replace=False
        )

        return [self.mol_dataset[idx] for idx in exemplar_idxs]

    def get_random_candidates(self):
        targets = self.mol_dataset.Y.squeeze()
        candidate_idxs = np.random.choice(
            targets.shape[0], size=(self.n_candidates,), replace=False
        )
        return [self.mol_dataset[idx] for idx in candidate_idxs]

    def __getitem__(self, idx) -> RandomPairDataPoint:
        return RandomPairDataPoint(
            self.mol_dataset[idx],
            (self.get_exemplar_candidates() + self.get_random_candidates()),
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


class RandomPairDataModule(L.LightningDataModule):
    def __init__(self, mol_ds_train, mol_ds_val) -> None:
        super().__init__()
        self.mol_ds_train: data.MoleculeDataset = mol_ds_train
        self.mol_ds_val: data.MoleculeDataset = mol_ds_val
        self.batch_size = 32
        self.candidate_size = 8

        self.ds_train = None
        self.ds_val = None

    def setup(self, stage=None):
        self.ds_train = RandomPairDataset(self.mol_ds_train, self.candidate_size)
        self.ds_val = RandomPairDataset(self.mol_ds_val, self.candidate_size)

    def train_dataloader(self):
        assert self.ds_train is not None
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=RandomPairDataset.collate_function,
            worker_init_fn=seed_worker,
            num_workers=8,
        )

    def val_dataloader(self):
        assert self.ds_val is not None
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=RandomPairDataset.collate_function,
            num_workers=8,
        )
