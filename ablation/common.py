"""Shared helpers for the ablation studies.

All ablation studies log to a single wandb project (pass `--wandb-cf.project-name
ablations`), with one named run per study and every sweep point logged as a history
step in that run. This module holds the split helpers, the wandb init/artifact
plumbing, and the generic per-sweep-point train+log routine that the studies in
`studies.py` reuse.
"""

import pickle
from dataclasses import asdict
from datetime import datetime

import numpy as np

from config import SplitType, TrainConfig, WandbConfig, WandbEnabled
from evaluate.train import get_group_splitters, get_random_splitters


def nested_stratified_fractions(df, fractions, seed):
    """Yield (fraction, sub_df) where each smaller fraction is a stratified subset
    of every larger one (20% subset 40% subset ... subset 100%).

    Within each `bin_target` class we shuffle the row indices once with a seeded
    RNG, then for each fraction take the leading prefix of that class. Because the
    per-class ordering is fixed, prefixes nest; because we slice each class by the
    same fraction, the class balance is preserved.
    """
    rng = np.random.RandomState(seed)

    class_indices = {}
    for label, group in df.groupby("bin_target"):
        idxs = group.index.to_numpy()
        rng.shuffle(idxs)
        class_indices[label] = idxs

    for fraction in sorted(fractions):
        selected = []
        for idxs in class_indices.values():
            n = round(len(idxs) * fraction)
            selected.append(idxs[:n])

        sub_idxs = np.concatenate(selected)
        sub_df = df.loc[sub_idxs].reset_index(drop=True)
        yield fraction, sub_df


def single_split(df, n_splits, seed, split_type):
    """Build one train/val/test split (no cross-validation) for the given split type.

    Mirrors the first iteration of `generate_repeated_5xn_splits`: the outer
    splitter yields train vs val+test (ratio 1/n_splits), then the inner 2-fold
    splitter halves val+test into val and test. For SCAFFOLD/BUTINA the grouping
    column keeps clusters disjoint across the split; RANDOM ignores groups.
    """
    if split_type == SplitType.RANDOM:
        outer_splitter, inner_splitter = get_random_splitters(seed, n_outer=n_splits)
        groups_of = lambda _df: None  # noqa: E731
    elif split_type == SplitType.SCAFFOLD:
        outer_splitter, inner_splitter = get_group_splitters(seed, n_outer=n_splits)
        groups_of = lambda _df: _df["scaffold_cluster"]  # noqa: E731
    elif split_type == SplitType.BUTINA:
        outer_splitter, inner_splitter = get_group_splitters(seed, n_outer=n_splits)
        groups_of = lambda _df: _df["butina_cluster"]  # noqa: E731
    else:
        raise ValueError(split_type)

    train_idxs, val_test_idxs = next(
        outer_splitter.split(df, y=df["bin_target"], groups=groups_of(df))
    )
    train_df = df.loc[train_idxs].reset_index(drop=True)
    val_test_df = df.loc[val_test_idxs].reset_index(drop=True)

    val_idxs, test_idxs = next(
        inner_splitter.split(
            val_test_df,
            y=val_test_df["bin_target"],
            groups=groups_of(val_test_df),
        )
    )
    val_df = val_test_df.loc[val_idxs].reset_index(drop=True)
    test_df = val_test_df.loc[test_idxs].reset_index(drop=True)

    return train_df, val_df, test_df


def init_ablation_run(wandb_cf: WandbConfig, run_name: str, extra_tags):
    """Start the single named wandb run for an ablation study (no-op if disabled).

    All studies tag with ``"ablation"`` plus the study-specific `extra_tags`. The
    run name is `run_name` with a ``_YYYYmmdd_HHMMSS`` timestamp appended, so each
    re-run is a distinct, identifiable run sharing the study prefix; the plot
    notebooks select the latest run matching that prefix.
    """
    if not isinstance(wandb_cf, WandbEnabled):
        return None

    import wandb

    wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
    tags = set(wandb_cf.tags) | {"ablation", *extra_tags}

    timestamped_name = f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=wandb_cf.project_name, name=timestamped_name, tags=list(tags)
    )
    run.mark_preempting()
    return run


def log_artifacts(label, model_name, predictions, split):
    """Log predictions + split as a wandb artifact, named by the sweep `label`."""
    import wandb

    artifact = wandb.Artifact(  # type: ignore
        name=f"{label}_{model_name}_artifacts",
        type="generic",
    )

    with artifact.new_file("predictions.pkl", mode="wb") as f:
        test_pred_probs, test_preds = predictions["test"]
        val_pred_probs, val_preds = predictions["val"]
        pickle.dump(
            {
                "test_pred_probs": test_pred_probs,
                "test_preds": test_preds,
                "val_pred_probs": val_pred_probs,
                "val_preds": val_preds,
            },
            f,
        )

    with artifact.new_file("split.pkl", mode="wb") as f:
        train_df, val_df, test_df = split
        pickle.dump({"train": train_df, "cal": val_df, "test": test_df}, f)

    wandb.run.log_artifact(artifact)  # type: ignore


def evaluate_and_log(
    *,
    split,
    df_classification_threshold,
    model_class,
    model_cf,
    model_name: str,
    train_cf: TrainConfig,
    wandb_cf: WandbConfig,
    extra_cols: dict,
    label: str,
):
    """Train on one split, assemble a result row, and log it to the active run.

    `extra_cols` carries the study-specific columns (sweep value, ``n_train``,
    ``model``, ``dataset``); the row is `extra_cols | metrics | model_cf | train_cf`.
    When wandb is enabled the row is logged as a history step and the predictions +
    split are attached as an artifact keyed by `label`.
    """
    from evaluate.train import train_and_evaluate_split

    train_df, val_df, test_df = split
    metrics_dict, predictions = train_and_evaluate_split(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        df_classification_threshold=df_classification_threshold,
        model_class=model_class,
        model_config=model_cf,
        train_config=train_cf,
    )

    row = extra_cols | metrics_dict | asdict(model_cf) | asdict(train_cf)

    if isinstance(wandb_cf, WandbEnabled):
        import wandb

        wandb.log(row)  # type: ignore
        log_artifacts(label, model_name, predictions, split)

    print(row)
    return row
