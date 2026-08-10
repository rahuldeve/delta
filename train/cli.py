"""Train a single split, with losses and gradients streamed to wandb.

Same tyro surface as `evaluate.cli` — a model subcommand plus `--train-cf.*`,
`--model-cf.*` and `--wandb-cf.*` — but it fits one train/val/test split instead of
the repeated 5xn cross-validation, and it attaches a `WandbLogger` to the Lightning
trainer so the loss curves and gradient statistics are actually recorded. Use it to
inspect a model's training dynamics; use `evaluate.cli` to measure it.

Only the Lightning-backed models are exposed: xgboost has no per-step loss or
gradients to log, so it stays on the `evaluate.cli` path.
"""

from dataclasses import asdict

import tyro

from config import (
    LogConfig,
    TrainConfig,
    WandbConfig,
    WandbDisabled,
    WandbEnabled,
)
from data import SupportedDatasets
from data.prepare import prepare_dataset
from models.config import ChempropConfig, DeltapropConfig


def init_logging(
    *,
    wandb_cf: WandbConfig,
    log_cf: LogConfig,
    model_name: str,
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
):
    """Start the wandb run and build the trainer logger + instrumentation callbacks.

    Returns `(run, trainer_logger, callbacks)`, all inert when wandb is disabled. Tags
    mirror `evaluate.cli`'s so runs stay filterable the same way, plus `single-split`
    to keep these apart from the cross-validation runs in a shared project.
    """
    if not isinstance(wandb_cf, WandbEnabled):
        return None, None, []

    import wandb
    from lightning.pytorch.loggers import WandbLogger

    from train.callbacks import GradNormLogger, WandbWatch

    wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
    tags = set(wandb_cf.tags) | {
        model_name,
        dataset.name.lower(),
        train_cf.split_type,
        "single-split",
    }

    if train_cf.use_feats:
        tags = tags | {"with-feats"}

    run = wandb.init(project=wandb_cf.project_name, tags=list(tags))
    run.mark_preempting()

    callbacks = []
    if log_cf.grad_norm_every_n_steps:
        callbacks.append(GradNormLogger(every_n_steps=log_cf.grad_norm_every_n_steps))
    if log_cf.watch_log_freq:
        callbacks.append(
            WandbWatch(log_mode=log_cf.watch_log, log_freq=log_cf.watch_log_freq)
        )

    return run, WandbLogger(experiment=run), callbacks


def train_one_split(
    *,
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
    model_cf,
    model_class,
    model_name: str,
    wandb_cf: WandbConfig,
    log_cf: LogConfig,
    fold: int,
    drop_nan_features: bool = True,
):
    from train.core import single_split, train_and_evaluate_split

    run, trainer_logger, callbacks = init_logging(
        wandb_cf=wandb_cf,
        log_cf=log_cf,
        model_name=model_name,
        dataset=dataset,
        train_cf=train_cf,
    )

    df, df_classification_threshold = prepare_dataset(
        dataset, use_features=train_cf.use_feats, drop_nan_features=drop_nan_features
    )
    train_df, val_df, test_df = single_split(
        df, train_cf.n_splits, train_cf.random_seed, train_cf.split_type, fold=fold
    )

    metrics_dict, _predictions = train_and_evaluate_split(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        df_classification_threshold=df_classification_threshold,
        model_class=model_class,
        model_config=model_cf,
        train_config=train_cf,
        trainer_logger=trainer_logger,
        extra_callbacks=callbacks,
    )

    if run is not None:
        # The WandbLogger owns this run's step counter (it logs at
        # trainer.global_step), so the final numbers go to config/summary rather than
        # through wandb.log — a manual log here would be rejected as non-monotonic.
        run.config.update(
            asdict(model_cf)
            | asdict(train_cf)
            | dict(
                dataset=dataset.name,
                model=model_name,
                fold=fold,
                n_train=len(train_df),
                n_val=len(val_df),
                n_test=len(test_df),
            )
        )
        run.summary.update(metrics_dict)
        run.finish()

    print(metrics_dict)
    return metrics_dict


def chemprop(
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
    model_cf: ChempropConfig,
    log_cf: LogConfig = LogConfig(),
    fold: int = 0,
    # Keep the wandb union last: tyro renders it as a trailing subcommand, so any
    # parameter declared after it would have to be passed nested underneath it.
    wandb_cf: WandbConfig = WandbDisabled(),
):
    """Train chemprop on one split of `dataset`."""
    from models.chemprop_bl import ChempropRef

    return train_one_split(
        dataset=dataset,
        train_cf=train_cf,
        model_cf=model_cf,
        model_class=ChempropRef,
        model_name="chemprop",
        wandb_cf=wandb_cf,
        log_cf=log_cf,
        fold=fold,
    )


def deltaprop(
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
    model_cf: DeltapropConfig,
    log_cf: LogConfig = LogConfig(),
    fold: int = 0,
    # See `chemprop`: the wandb union must stay the last parameter.
    wandb_cf: WandbConfig = WandbDisabled(),
):
    """Train deltaprop on one split of `dataset`."""
    from models.deltaprop import DeltapropRef

    return train_one_split(
        dataset=dataset,
        train_cf=train_cf,
        model_cf=model_cf,
        model_class=DeltapropRef,
        model_name="deltaprop",
        wandb_cf=wandb_cf,
        log_cf=log_cf,
        fold=fold,
    )


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        dict(
            chemprop=chemprop,
            deltaprop=deltaprop,
        )
    )
