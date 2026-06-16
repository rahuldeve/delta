import pickle
from dataclasses import asdict

import tyro

from config import SplitType, TrainConfig, WandbConfig, WandbDisabled, WandbEnabled
from data import SupportedDatasets
from evaluate.cli import prepare_dataset
from models.config import ChempropConfig, DeltapropConfig

# Reuse the SCAFFOLD-aware single split + stratified-sampling helpers from the
# data-fraction ablation.
from ablation.gsk_hepg2_dataset_ablation import (
    nested_stratified_fractions,
    single_split,
)


def candidate_log_artifacts(label, model_name, predictions, split):
    """Log predictions + split as a wandb artifact, named by the sweep label.

    Mirrors `ablation.gsk_hepg2_ablation.ablation_log_artifacts` but keys the
    artifact name on the candidate-size sweep (e.g. ``cand16`` / ``baseline``)
    rather than the data fraction.
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
    candidate_sizes: tuple[int, ...] = tuple(range(4, 52, 4)),
):
    from evaluate.train import train_and_evaluate_split
    from models.chemprop_bl import ChempropRef
    from models.deltaprop import DeltapropRef

    # This ablation holds the data fixed and sweeps deltaprop's candidate-pool
    # size under the SCAFFOLD split, in the feature-free (graph-only) setting.
    # Pin both so the logged configs reflect reality regardless of CLI input.
    train_cf.use_feats = False
    train_cf.split_type = SplitType.SCAFFOLD

    if isinstance(wandb_cf, WandbEnabled):
        import wandb

        wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
        tags = set(wandb_cf.tags) | set(
            [
                "ablation",
                "db_malaria",
                "candidate_size",
                train_cf.split_type,
            ]
        )

        run_ = wandb.init(project=wandb_cf.project_name, tags=list(tags))
        run_.mark_preempting()

    df, df_classification_threshold = prepare_dataset(
        SupportedDatasets.DB_MALARIA,
        use_features=False,
        drop_nan_features=True,
    )

    # One SCAFFOLD split on the subsampled dataset, shared by every run below.
    split = single_split(
        df, train_cf.n_splits, train_cf.random_seed, train_cf.split_type
    )
    train_df, val_df, test_df = split

    def evaluate_and_log(model_class, model_cf, model_name, candidate_size, label):
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
                "candidate_size": candidate_size,
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
            candidate_log_artifacts(label, model_name, predictions, split)

        print(row)

    # chemprop is invariant to candidate_size; run it once as a reference line.
    evaluate_and_log(
        ChempropRef, chemprop_cf, "chemprop", candidate_size=None, label="baseline"
    )

    # deltaprop: sweep the candidate-pool size.
    for candidate_size in candidate_sizes:
        deltaprop_cf.candidate_size = candidate_size
        evaluate_and_log(
            DeltapropRef,
            deltaprop_cf,
            "deltaprop",
            candidate_size=candidate_size,
            label=f"cand{candidate_size}",
        )

    return None


if __name__ == "__main__":
    tyro.cli(run)
