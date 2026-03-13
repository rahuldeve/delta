import random

import numpy as np
import torch
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tanimoto_similarity_matrix(
    mols_a: list,
    mols_b: list,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Calculate Tanimoto similarity between two lists of RDKit molecules.

    Args:
        mols_a: First list of RDKit Mol objects.
        mols_b: Second list of RDKit Mol objects.
        radius: Morgan fingerprint radius (default: 2).
        n_bits: Number of bits in the fingerprint (default: 2048).

    Returns:
        A (len(mols_a), len(mols_b)) numpy array of Tanimoto similarities.
    """
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    def to_fp(mol):
        if mol is None:
            raise ValueError("Invalid molecule (None) in input list.")
        return generator.GetFingerprint(mol)

    fps_a = [to_fp(m) for m in mols_a]
    fps_b = [to_fp(m) for m in mols_b]

    similarity_matrix = np.zeros((len(fps_a), len(fps_b)))

    for i, fp_a in enumerate(fps_a):
        similarity_matrix[i] = DataStructs.BulkTanimotoSimilarity(fp_a, fps_b)

    return similarity_matrix
