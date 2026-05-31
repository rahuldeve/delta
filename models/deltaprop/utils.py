import torch
from chemprop.data import MoleculeDataset, collate_batch
from pytorch_lightning.utilities import move_data_to_device


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
        num_workers=8
    )
    all_embeds = []
    for batch in dl:
        batch = move_data_to_device(batch, model.device)
        res = model.embed_simple_batch(batch)
        all_embeds.append(res["embeds"])

    all_embeds = torch.cat(all_embeds)
    return all_embeds


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
        num_workers=8
    )
    all_scores = []
    for batch in dl:
        batch = move_data_to_device(batch, model.device)
        res = model.embed_simple_batch(batch)
        scores = model.interaction.projector(res["embeds"])
        all_scores.append(scores)

    all_scores = torch.cat(all_scores)
    return all_scores
