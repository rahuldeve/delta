import pickle
from dataclasses import asdict

import numpy as np
import tyro

from config import TrainConfig, WandbConfig, WandbDisabled, WandbEnabled
from data import SupportedDatasets
from evaluate.cli import prepare_dataset
from evaluate.train import get_group_splitters
from models.config import ChempropConfig, DeltapropConfig


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


def single_butina_split(df, n_splits, seed):
    """Build one Butina-grouped train/val/test split (no cross-validation).

    Mirrors the first iteration of `generate_repeated_5xn_splits`: the outer
    StratifiedGroupKFold yields train vs val+test (ratio 1/n_splits), then the
    inner 2-fold splitter halves val+test into val and test. Groups are the
    precomputed `butina_cluster` column, so no cluster leaks across the split.
    """
    outer_splitter, inner_splitter = get_group_splitters(seed, n_outer=n_splits)

    train_idxs, val_test_idxs = next(
        outer_splitter.split(df, y=df["bin_target"], groups=df["butina_cluster"])
    )
    train_df = df.loc[train_idxs].reset_index(drop=True)
    val_test_df = df.loc[val_test_idxs].reset_index(drop=True)

    val_idxs, test_idxs = next(
        inner_splitter.split(
            val_test_df,
            y=val_test_df["bin_target"],
            groups=val_test_df["butina_cluster"],
        )
    )
    val_df = val_test_df.loc[val_idxs].reset_index(drop=True)
    test_df = val_test_df.loc[test_idxs].reset_index(drop=True)

    return train_df, val_df, test_df


def ablation_log_artifacts(fraction, model_name, predictions, split):
    import wandb

    artifact = wandb.Artifact(  # type: ignore
        name=f"frac{int(fraction * 100)}_{model_name}_artifacts",
        type="generic",
    )

    with artifact.new_file("predictions.pkl", mode="wb") as f:
        pred_probs, preds = predictions
        pickle.dump({"pred_probs": pred_probs, "preds": preds}, f)

    with artifact.new_file("split.pkl", mode="wb") as f:
        train_df, val_df, test_df = split
        pickle.dump({"train": train_df, "cal": val_df, "test": test_df}, f)

    wandb.run.log_artifact(artifact)  # type: ignore


def run(
    train_cf: TrainConfig,
    chemprop_cf: ChempropConfig,
    deltaprop_cf: DeltapropConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
):
    from evaluate.train import train_and_evaluate_split
    from models.chemprop_bl import ChempropRef
    from models.deltaprop import DeltapropRef

    # This ablation only studies the feature-free (graph-only) setting for now;
    # pin the flag so logged configs reflect reality regardless of CLI input.
    train_cf.use_feats = False

    if isinstance(wandb_cf, WandbEnabled):
        import wandb

        wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
        tags = set(wandb_cf.tags) | set(
            [
                "ablation",
                "gsk_hepg2",
                train_cf.split_type,
            ]
        )

        run_ = wandb.init(project=wandb_cf.project_name, tags=list(tags))
        run_.mark_preempting()

    # This ablation only studies the feature-free (graph-only) setting for now.
    df, df_classification_threshold = prepare_dataset(
        SupportedDatasets.GSK_HEPG2,
        use_features=False,
        drop_nan_features=True,
    )

    models = [
        ("chemprop", ChempropRef, chemprop_cf),
        ("deltaprop", DeltapropRef, deltaprop_cf),
    ]

    for fraction, sub_df in nested_stratified_fractions(
        df, fractions, train_cf.random_seed
    ):
        split = single_butina_split(sub_df, train_cf.n_splits, train_cf.random_seed)
        train_df, val_df, test_df = split

        for model_name, model_class, model_cf in models:
            metrics_dict, predictions = train_and_evaluate_split(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                df_classification_threshold=df_classification_threshold,
                model_class=model_class,
                model_config=model_cf,
                train_config=train_cf,
            )

            row = (
                {
                    "fraction": fraction,
                    "n_train": len(train_df),
                    "model": model_name,
                }
                | metrics_dict
                | asdict(model_cf)
                | asdict(train_cf)
                | dict(dataset="GSK_HEPG2")
            )

            if isinstance(wandb_cf, WandbEnabled):
                wandb.log(row)  # type: ignore
                ablation_log_artifacts(fraction, model_name, predictions, split)

            print(row)

    return None


if __name__ == "__main__":
    tyro.cli(run)
