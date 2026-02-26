import sys

sys.path.append("..")

from dataclasses import asdict
from enum import Enum, auto

import tyro
from config import TrainConfig, WandbConfig, WandbDisabled, WandbEnabled

from models.config import BaselineConfig, DeltapropConfig


class SupportedDatasets(Enum):
    SINGLE_TARGET_TBA = auto()
    DUAL_TARGET_TBA = auto()
    GSK_HEPG2 = auto()
    PK = auto()
    DB_MALARIA = auto()
    DB_HEPG2 = auto()


def prepare_dataset(dataset: SupportedDatasets):
    # Lazy import here to prevent cli startup from being slow
    import ray
    from data import (
        load_dual_target_tba,
        load_gsk_hepg2,
        load_single_target_tba,
        load_pk,
        load_derbyshire_malaria,
        load_derbyshire_hepg2
    )

    ray.init(ignore_reinit_error=True, num_cpus=4, runtime_env={"working_dir": "../"})

    if dataset == SupportedDatasets.SINGLE_TARGET_TBA:
        df, df_classification_threshold = load_single_target_tba()
    elif dataset == SupportedDatasets.DUAL_TARGET_TBA:
        df, df_classification_threshold = load_dual_target_tba()
    elif dataset == SupportedDatasets.GSK_HEPG2:
        df, df_classification_threshold = load_gsk_hepg2()
    elif dataset == SupportedDatasets.PK:
        df, df_classification_threshold = load_pk()
    elif dataset == SupportedDatasets.DB_MALARIA:
        df, df_classification_threshold = load_derbyshire_malaria()
    elif dataset == SupportedDatasets.DB_HEPG2:
        df, df_classification_threshold = load_derbyshire_hepg2()
    else:
        raise ValueError(dataset)

    ray.shutdown()

    return df, df_classification_threshold


def baseline(
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
    model_cf: BaselineConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
):
    from train import train_and_evaluate

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
                | dict(dataset=dataset, model="deltaprop")
            )

        print(result_dict)

    return None


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(dict(baseline=baseline, deltaprop=deltaprop))
