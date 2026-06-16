import pickle
from dataclasses import asdict

import tyro

from config import SplitType, TrainConfig, WandbConfig, WandbDisabled, WandbEnabled
from data import SupportedDatasets
from evaluate.cli import prepare_dataset
from models.config import ChempropConfig, DeltapropConfig

# Reuse the SCAFFOLD-aware single split helper from the data-fraction ablation.
from ablation.gsk_hepg2_dataset_ablation import single_split


def hard_neg_log_artifacts(label, model_name, predictions, split):
    """Log predictions + split as a wandb artifact, named by the sweep label.

    Mirrors the candidate/dataset ablation helpers but keys the artifact name on
    the hard-negative fraction sweep (e.g. ``frac0.4`` / ``baseline``).
    """
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


def run(
    train_cf: TrainConfig,
    chemprop_cf: ChempropConfig,
    deltaprop_cf: DeltapropConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
    frac_hard_values: tuple[float, ...] = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
):
    from evaluate.train import train_and_evaluate_split
    from models.chemprop_bl import ChempropRef
    from models.deltaprop import DeltapropRef

    # This ablation holds the data fixed and sweeps the fraction of hard
    # negatives mined into each deltaprop training batch, under the SCAFFOLD
    # split, in the feature-free (graph-only) setting. Pin both so the logged
    # configs reflect reality regardless of CLI input.
    train_cf.use_feats = False
    train_cf.split_type = SplitType.SCAFFOLD

    if isinstance(wandb_cf, WandbEnabled):
        import wandb

        wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
        tags = set(wandb_cf.tags) | set(
            [
                "ablation",
                "db_malaria",
                "frac_hard",
                train_cf.split_type,
            ]
        )

        run_ = wandb.init(project=wandb_cf.project_name, tags=list(tags))
        run_.mark_preempting()

    # DB_MALARIA is small, so we use the full dataset (no subsampling).
    df, df_classification_threshold = prepare_dataset(
        SupportedDatasets.DB_MALARIA,
        use_features=False,
        drop_nan_features=True,
    )

    # One SCAFFOLD split shared by every run below.
    split = single_split(
        df, train_cf.n_splits, train_cf.random_seed, train_cf.split_type
    )
    train_df, val_df, test_df = split

    def evaluate_and_log(model_class, model_cf, model_name, frac_hard, label):
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
                "frac_hard": frac_hard,
                "n_train": len(train_df),
                "model": model_name,
            }
            | metrics_dict
            | asdict(model_cf)
            | asdict(train_cf)
            | dict(dataset="DB_MALARIA")
        )

        if isinstance(wandb_cf, WandbEnabled):
            import wandb

            wandb.log(row)  # type: ignore
            hard_neg_log_artifacts(label, model_name, predictions, split)

        print(row)

    # chemprop is invariant to frac_hard; run it once as a reference line.
    evaluate_and_log(
        ChempropRef, chemprop_cf, "chemprop", frac_hard=None, label="baseline"
    )

    # deltaprop: sweep the hard-negative fraction.
    for frac_hard in frac_hard_values:
        deltaprop_cf.frac_hard = frac_hard
        evaluate_and_log(
            DeltapropRef,
            deltaprop_cf,
            "deltaprop",
            frac_hard=frac_hard,
            label=f"frac{frac_hard}",
        )

    return None


if __name__ == "__main__":
    tyro.cli(run)
