import random

import lightning as L
import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers
from torch.utils.data import DataLoader, Dataset


class ShuffledPairsDataset(Dataset):
    def __init__(self, df, sample_ratio=5):
        super().__init__()
        self.df = df
        self.sample_ratio = sample_ratio
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.pairs: list = []
        self.mg_cache: list = []

        self.build_mg_cache()
        self.update_pairs()

    def build_mg_cache(self):
        self.mg_cache = self.df["mol"].map(self.featurizer).tolist()

    def update_pairs(self):
        N = len(self.df)

        weights = self.df["per_inhibition"].to_numpy()
        weights = np.where(weights >= 50, 2.0, 1.0)
        weights = weights / weights.sum()

        pairs = [
            (i, j)
            for i in range(N)
            for j in np.random.choice(
                len(self.df), size=(self.sample_ratio,), p=weights, replace=False
            )
        ]

        pairs += [(j, i) for i, j in pairs]
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        left_idx, right_idx = self.pairs[idx]
        left_mg, right_mg = self.mg_cache[left_idx], self.mg_cache[right_idx]
        delta = (
            self.df["per_inhibition"][left_idx] >= self.df["per_inhibition"][right_idx]
        ).astype(float)

        left_datum = data.datasets.Datum(
            left_mg, None, None, np.array([delta]), 1.0, None, None
        )

        right_datum = data.datasets.Datum(right_mg, None, None, None, 1.0, None, None)

        return [left_datum, right_datum]


class ExemplarDataset(Dataset):
    def __init__(self, df_regular, df_exemplars) -> None:
        super().__init__()
        self.df_exemplars = df_exemplars
        self.df_regular = df_regular
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.pairs = []
        self.exemplar_mg_cache: list = []
        self.regular_mg_cache: list = []

        self.build_pairs()
        self.build_mg_cache()

    def build_mg_cache(self):
        self.exemplar_mg_cache = self.df_exemplars["mol"].map(self.featurizer).tolist()
        self.regular_mg_cache = self.df_regular["mol"].map(self.featurizer).tolist()

    def build_pairs(self):
        self.pairs = [
            (i, j)
            for i in range(len(self.df_regular))
            for j in range(len(self.df_exemplars))
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        regular_idx, exemplar_idx = self.pairs[idx]
        regular_mol = self.regular_mg_cache[regular_idx]
        exemplar_mol = self.exemplar_mg_cache[exemplar_idx]
        delta = (
            self.df_regular["per_inhibition"][regular_idx]
            >= self.df_exemplars["per_inhibition"][exemplar_idx]
        ).astype(float)

        regular_datum = data.datasets.Datum(
            regular_mol, None, None, np.array([delta]), 1.0, None, None
        )

        exemplar_datum = data.datasets.Datum(
            exemplar_mol, None, None, None, 1.0, None, None
        )

        return [regular_datum, exemplar_datum]


# see https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ConstrastiveDataModule(L.LightningDataModule):
    def __init__(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        train_dataset = ShuffledPairsDataset(self.df_train, sample_ratio=20)
        return DataLoader(
            dataset=train_dataset,
            batch_size=2048,
            shuffle=True,
            collate_fn=data.dataloader.collate_multicomponent,
            worker_init_fn=seed_worker,
            num_workers=12,
        )

    def val_dataloader(self):
        # exemplar_df = pd.concat([
        #     df_train[df_train['per_inhibition'] >= -15].sample(100),
        #     df_train[df_train['per_inhibition'] < -15].sample(50)
        # ]).reset_index(drop=True)

        val_dataset = ShuffledPairsDataset(self.df_val, sample_ratio=20)
        return DataLoader(
            dataset=val_dataset,
            batch_size=2048,
            shuffle=False,
            collate_fn=data.dataloader.collate_multicomponent,
            worker_init_fn=seed_worker,
            num_workers=12,
        )
