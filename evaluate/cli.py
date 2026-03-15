from dataclasses import asdict

import tyro

from config import TrainConfig, WandbConfig, WandbDisabled, WandbEnabled
from data import SupportedDatasets
from models.config import BaselineConfig, DeltapropConfig


def prepare_dataset(dataset: SupportedDatasets):
    # Lazy import here to prevent cli startup from being slow
    import ray

    from data.loaders import load_dataset
    from data.preprocessing import preprocess_ray

    ray.init(ignore_reinit_error=True, num_cpus=4)

    df, df_classification_threshold = load_dataset(dataset)
    df = preprocess_ray(df)

    ray.shutdown()

    return df, df_classification_threshold


def baseline(
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
    model_cf: BaselineConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
):
    from evaluate.train import train_and_evaluate
    from models import baseline

    if isinstance(wandb_cf, WandbEnabled):
        import wandb

        wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
        tags = set(wandb_cf.tags) | set(['baseline', dataset.name.lower(), train_cf.split_type])
        run = wandb.init(project=wandb_cf.project_name, tags=list(tags))
        run.mark_preempting()

    df, df_classification_threshold = prepare_dataset(dataset)
    result_iter = train_and_evaluate(
        df=df,
        df_classification_threshold=df_classification_threshold,
        model_module=baseline,
        model_config=model_cf,
        train_config=train_cf,
    )

    for result_dict in result_iter:
        if isinstance(wandb_cf, WandbEnabled):
            wandb.log(  # type: ignore
                result_dict
                | asdict(model_cf)
                | asdict(train_cf)
                | dict(dataset=dataset, model="baseline")
            )

        print(result_dict)

    return None


def deltaprop(
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
    model_cf: DeltapropConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
):
    from evaluate.train import train_and_evaluate
    from models import deltaprop

    if isinstance(wandb_cf, WandbEnabled):
        import wandb

        wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
        tags = set(wandb_cf.tags) | set(['deltaprop', dataset.name.lower(), train_cf.split_type])
        run = wandb.init(project=wandb_cf.project_name, tags=list(tags))
        run.mark_preempting()

    df, df_classification_threshold = prepare_dataset(dataset)
    result_iter = train_and_evaluate(
        df=df,
        df_classification_threshold=df_classification_threshold,
        model_module=deltaprop,
        model_config=model_cf,
        train_config=train_cf,
    )

    for result_dict in result_iter:
        if isinstance(wandb_cf, WandbEnabled):
            wandb.log(  # type: ignore
                result_dict
                | asdict(model_cf)
                | asdict(train_cf)
                | dict(dataset=dataset, model="deltaprop-btl")
            )

        print(result_dict)

    return None


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(dict(baseline=baseline, deltaprop=deltaprop))
