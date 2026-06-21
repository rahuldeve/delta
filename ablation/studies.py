"""Ablation studies, exposed as subcommands of `python -m ablation`.

Each study logs to the shared `ABLATION_PROJECT` wandb project as a single run with
a stable custom name; every sweep point is a history step within that run. The
shared split / wandb / train+log plumbing lives in `ablation.common`.
"""

from config import SplitType, TrainConfig, WandbConfig, WandbDisabled
from data import SupportedDatasets
from evaluate.cli import prepare_dataset
from models.config import ChempropConfig, DeltapropConfig

from ablation.common import (
    evaluate_and_log,
    init_ablation_run,
    nested_stratified_fractions,
    single_split,
)


def candidate_size(
    train_cf: TrainConfig,
    chemprop_cf: ChempropConfig,
    deltaprop_cf: DeltapropConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
    dataset: SupportedDatasets = SupportedDatasets.DB_MALARIA,
    candidate_sizes: tuple[int, ...] = tuple(range(8, 52, 8)),
    seeds: tuple[int, ...] = (42, 53, 64, 75, 86),
    run_name: str | None = None,
):
    """Sweep deltaprop's candidate-pool size on `dataset` (SCAFFOLD, graph-only).

    The full sweep is repeated once per seed in `seeds`; each seed re-randomizes both
    the split and model init, so every sweep point gets one measurement per seed.
    """
    from models.chemprop_bl import ChempropRef
    from models.deltaprop import DeltapropRef

    # Default the run name to the dataset so per-dataset runs stay distinguishable;
    # the plot notebooks select the latest run matching this prefix.
    if run_name is None:
        run_name = f"{dataset.name.lower()}_candidate_size"

    # Hold the data fixed and sweep the candidate-pool size under the SCAFFOLD
    # split, feature-free. Pin both so the logged configs reflect reality.
    train_cf.use_feats = False
    train_cf.split_type = SplitType.SCAFFOLD

    init_ablation_run(
        wandb_cf,
        run_name,
        extra_tags={dataset.name.lower(), "candidate_size", train_cf.split_type},
    )

    df, df_classification_threshold = prepare_dataset(
        dataset,
        use_features=False,
        drop_nan_features=True,
    )

    for seed in seeds:
        train_cf.random_seed = seed

        # One SCAFFOLD split (per seed) shared by every run below.
        split = single_split(
            df, train_cf.n_splits, train_cf.random_seed, train_cf.split_type
        )
        train_df = split[0]

        def log(model_class, model_cf, model_name, candidate_size, label):
            evaluate_and_log(
                split=split,
                df_classification_threshold=df_classification_threshold,
                model_class=model_class,
                model_cf=model_cf,
                model_name=model_name,
                train_cf=train_cf,
                wandb_cf=wandb_cf,
                extra_cols={
                    "candidate_size": candidate_size,
                    "seed": seed,
                    "n_train": len(train_df),
                    "model": model_name,
                    "dataset": dataset.name,
                },
                label=label,
            )

        # chemprop is invariant to candidate_size; run it once as a reference line.
        log(
            ChempropRef,
            chemprop_cf,
            "chemprop",
            candidate_size=None,
            label=f"baseline_seed{seed}",
        )

        # deltaprop: sweep the candidate-pool size.
        for candidate_size in candidate_sizes:
            deltaprop_cf.candidate_size = candidate_size
            log(
                DeltapropRef,
                deltaprop_cf,
                "deltaprop",
                candidate_size=candidate_size,
                label=f"cand{candidate_size}_seed{seed}",
            )

    return None


def frac_hard(
    train_cf: TrainConfig,
    chemprop_cf: ChempropConfig,
    deltaprop_cf: DeltapropConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
    dataset: SupportedDatasets = SupportedDatasets.DB_MALARIA,
    frac_hard_values: tuple[float, ...] = (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ),
    seeds: tuple[int, ...] = (42, 53, 64, 75, 86),
    run_name: str | None = None,
):
    """Sweep the hard-negative mining fraction on `dataset` (SCAFFOLD, graph-only).

    The full sweep is repeated once per seed in `seeds`; each seed re-randomizes both
    the split and model init, so every sweep point gets one measurement per seed.
    """
    from models.chemprop_bl import ChempropRef
    from models.deltaprop import DeltapropRef

    # Default the run name to the dataset so per-dataset runs stay distinguishable;
    # the plot notebooks select the latest run matching this prefix.
    if run_name is None:
        run_name = f"{dataset.name.lower()}_frac_hard"

    # Hold the data fixed and sweep the fraction of hard negatives mined into each
    # deltaprop batch under the SCAFFOLD split, feature-free. Pin both so the
    # logged configs reflect reality.
    train_cf.use_feats = False
    train_cf.split_type = SplitType.SCAFFOLD

    init_ablation_run(
        wandb_cf,
        run_name,
        extra_tags={dataset.name.lower(), "frac_hard", train_cf.split_type},
    )

    # The dataset is used in full (no subsampling).
    df, df_classification_threshold = prepare_dataset(
        dataset,
        use_features=False,
        drop_nan_features=True,
    )

    for seed in seeds:
        train_cf.random_seed = seed

        # One SCAFFOLD split (per seed) shared by every run below.
        split = single_split(
            df, train_cf.n_splits, train_cf.random_seed, train_cf.split_type
        )
        train_df = split[0]

        def log(model_class, model_cf, model_name, frac_hard, label):
            evaluate_and_log(
                split=split,
                df_classification_threshold=df_classification_threshold,
                model_class=model_class,
                model_cf=model_cf,
                model_name=model_name,
                train_cf=train_cf,
                wandb_cf=wandb_cf,
                extra_cols={
                    "frac_hard": frac_hard,
                    "seed": seed,
                    "n_train": len(train_df),
                    "model": model_name,
                    "dataset": dataset.name,
                },
                label=label,
            )

        # chemprop is invariant to frac_hard; run it once as a reference line.
        log(
            ChempropRef,
            chemprop_cf,
            "chemprop",
            frac_hard=None,
            label=f"baseline_seed{seed}",
        )

        # deltaprop: sweep the hard-negative fraction.
        for frac_hard in frac_hard_values:
            deltaprop_cf.frac_hard = frac_hard
            log(
                DeltapropRef,
                deltaprop_cf,
                "deltaprop",
                frac_hard=frac_hard,
                label=f"frac{frac_hard}_seed{seed}",
            )

    return None


def gsk_hepg2_data_fraction(
    train_cf: TrainConfig,
    chemprop_cf: ChempropConfig,
    deltaprop_cf: DeltapropConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: tuple[int, ...] = (42, 53, 64, 75, 86),
    run_name: str = "gsk_hepg2_data_fraction",
):
    """Sweep the training-data fraction on GSK_HEPG2 for chemprop vs deltaprop.

    The full sweep is repeated once per seed in `seeds`; each seed re-randomizes the
    fraction subsampling, the split, and model init, so every sweep point gets one
    measurement per seed.
    """
    from models.chemprop_bl import ChempropRef
    from models.deltaprop import DeltapropRef

    # This ablation only studies the feature-free (graph-only) setting for now;
    # pin the flag so logged configs reflect reality regardless of CLI input.
    train_cf.use_feats = False

    init_ablation_run(
        wandb_cf,
        run_name,
        extra_tags={"gsk_hepg2", "data_fraction", train_cf.split_type},
    )

    df, df_classification_threshold = prepare_dataset(
        SupportedDatasets.GSK_HEPG2,
        use_features=False,
        drop_nan_features=True,
    )

    models = [
        ("chemprop", ChempropRef, chemprop_cf),
        ("deltaprop", DeltapropRef, deltaprop_cf),
    ]

    for seed in seeds:
        train_cf.random_seed = seed

        for fraction, sub_df in nested_stratified_fractions(
            df, fractions, train_cf.random_seed
        ):
            split = single_split(
                sub_df, train_cf.n_splits, train_cf.random_seed, train_cf.split_type
            )
            train_df = split[0]

            for model_name, model_class, model_cf in models:
                evaluate_and_log(
                    split=split,
                    df_classification_threshold=df_classification_threshold,
                    model_class=model_class,
                    model_cf=model_cf,
                    model_name=model_name,
                    train_cf=train_cf,
                    wandb_cf=wandb_cf,
                    extra_cols={
                        "fraction": fraction,
                        "seed": seed,
                        "n_train": len(train_df),
                        "model": model_name,
                        "dataset": "GSK_HEPG2",
                    },
                    label=f"frac{int(fraction * 100)}_seed{seed}",
                )

    return None
