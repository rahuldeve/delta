"""Full sweep: all datasets x all models x all splits.

Generated for orchestrator.py. Rules:
  - PK uses --train-cf.n-splits 2; every other dataset uses 5.
  - GSK_HEPG2 uses --train-cf.batch-size 256 for deltaprop and chemprop.
  - deltaprop on GSK_HEPG2 uses --model-cf.candidate-size 12.
  - chemprop/deltaprop are swept both with and without --train-cf.use-feats.
  - xgboost always runs with --train-cf.use-feats (it needs molecular descriptors).
  - feature runs are tagged --wandb-cf.model-name-suffix feat.

Run with:
    uv run --active python orchestrator.py --jobs-file jobs.py --dry-run
    uv run --active python orchestrator.py --jobs-file jobs.py
"""

WANDB_PROJECT = "evaluate_all_v6"

DATASETS = [
    "SINGLE_TARGET_TBA",
    "DUAL_TARGET_TBA",
    #"GSK_HEPG2",
    "PK",
    "DB_MALARIA",
    #"DB_HEPG2",
]
MODELS = ["deltaprop"]
SPLITS = ["SCAFFOLD"]


def build_command(model: str, dataset: str, split: str, use_feats: bool) -> str:
    parts = [
        "uv run --active python -m evaluate.cli",
        model,
        f"--dataset {dataset}",
        f"--train-cf.split-type {split}",
    ]

    # PK is tiny: fewer folds. Everything else uses the default 5.
    parts.append("--train-cf.n-splits 2" if dataset == "PK" else "--train-cf.n-splits 5")

    # Larger batch for the big dataset on the GPU models.
    if dataset == "GSK_HEPG2" and model in ("chemprop", "deltaprop"):
        parts.append("--train-cf.batch-size 256")

    if use_feats:
        parts.append("--train-cf.use-feats")

    # deltaprop pairwise-ranking candidate pool size (only needed on the big dataset).
    if model == "deltaprop" and dataset == "GSK_HEPG2":
        parts.append("--model-cf.candidate-size 12")

    parts.append(f"wandb-cf:wandb-enabled --wandb-cf.project-name {WANDB_PROJECT}")

    # Tag feature-based runs so they're distinguishable in wandb.
    if use_feats:
        parts.append("--wandb-cf.model-name-suffix feat")

    return " ".join(parts)


def feats_options(model: str) -> list[bool]:
    return [False]
    # xgboost always needs descriptors; the GPU models are swept both ways.
    if model == "xgboost":
        return [True]
    return [False, True]


JOBS = [
    build_command(model, dataset, split, use_feats)
    for dataset in DATASETS
    for model in MODELS
    for split in SPLITS
    for use_feats in feats_options(model)
]
