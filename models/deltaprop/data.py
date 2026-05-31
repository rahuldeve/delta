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


def build_discordancy_matrix(
    raw_scores: np.ndarray,
    reference_scores: np.ndarray,
    nu: float,
) -> np.ndarray:
    exp_s = np.exp(raw_scores)  # (N,)

    # (N, N) terms
    exp_i = exp_s[:, None]                              # exp(s_i)
    exp_j = exp_s[None, :]                              # exp(s_j)
    exp_tie = nu * np.exp((raw_scores[:, None] + raw_scores[None, :]) / 2)  # ν·exp((s_i+s_j)/2)

    # Net model preference for i over j, accounting for ν
    net_model_pref = (exp_i - exp_j) / (exp_i + exp_j + exp_tie)  # in (-1, 1)

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
    if total_discordancy == 0:
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
        discordancy_degree: np.ndarray | None = None,
    ):
        super().__init__()
        self.anchor_dataset = anchor_dataset
        self.candidate_dataset = candidate_dataset
        self.binary_threshold = binary_threshold
        self.n_candidates = n_candidates
        self.frac_hard = frac_hard
        self.discordancy_degree = discordancy_degree


    def __len__(self):
        return len(self.anchor_dataset)
    
    def get_hard_neg_candidates(self, idx: int, n: int):
        if self.discordancy_degree is None:
            return []
        
        if n == 0:
            return []
        
        selected_idxs = top_k_discordant(
            self.discordancy_degree, idx, n
        )
        
        return [self.candidate_dataset[idx] for idx in selected_idxs]
    

    def get_random_candidates(self, n_random: int):
        targets = self.candidate_dataset.Y.squeeze()
        if isinstance(self.binary_threshold, GT):
            pos_class_mask = targets >= self.binary_threshold.th
        elif isinstance(self.binary_threshold, LT):
            pos_class_mask = targets <= self.binary_threshold.th
        else:
            raise ValueError(self.binary_threshold)

        pos_class_idxs = np.argwhere(pos_class_mask).squeeze()
        neg_class_idxs = np.argwhere(~pos_class_mask).squeeze()

        pos_class_sample_count = min(int(0.8*n_random), pos_class_idxs.shape[0])
        random_pos_class_idxs = np.random.choice(
            pos_class_idxs,
            size=(pos_class_sample_count, ),
            replace=False,
        )
        random_pos_candidates = [self.candidate_dataset[idx] for idx in random_pos_class_idxs]

        neg_class_sample_count = min(n_random - pos_class_sample_count, pos_class_idxs.shape[0])
        random_neg_class_idxs = np.random.choice(
            neg_class_idxs,
            size=(neg_class_sample_count, ),
            replace=False,
        )
        random_neg_candidates = [self.candidate_dataset[idx] for idx in random_neg_class_idxs]

        return random_pos_candidates + random_neg_candidates

    def __getitem__(self, idx) -> RandomPairDataPoint:
        hard_neg_candidates = self.get_hard_neg_candidates(idx, int(self.frac_hard * self.n_candidates))
        random_candidates = self.get_random_candidates(self.n_candidates - len(hard_neg_candidates))
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
        num_workers: int = 8,
    ) -> None:
        super().__init__()

        self.train_ds = RandomPairDataset(
            anchor_dataset=train_mol_ds, 
            candidate_dataset=train_mol_ds, 
            binary_threshold=binary_threshold, 
            n_candidates=n_candidates
        )

        self.val_ds = RandomPairDataset(
            anchor_dataset=val_mol_ds, 
            candidate_dataset=train_mol_ds, 
            binary_threshold=binary_threshold, 
            n_candidates=n_candidates
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        if self.trainer is not None and self.trainer.current_epoch > 0:
            # assert self.trainer.model is not None
            train_mol_ds = self.train_ds.anchor_dataset
            model = self.trainer.model
            theta_hat_train = score_all(train_mol_ds, model).squeeze()

            with torch.no_grad():
                nu = model.interaction.log_nu.exp().cpu().item() # type: ignore

            self.update_discordancy_mat_train(
                theta_hat_train.cpu().numpy(),
                nu,
            )
            

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=RandomPairDataset.collate_function,
            worker_init_fn=seed_worker,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=RandomPairDataset.collate_function,
            num_workers=self.num_workers,
        )

    def update_discordancy_mat_train(self, train_model_scores, nu: float):
        reference_scores = self.train_ds.anchor_dataset.Y.squeeze()
        self.train_ds.discordancy_degree = build_discordancy_matrix(train_model_scores, reference_scores, nu)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


