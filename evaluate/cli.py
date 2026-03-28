import pickle
from dataclasses import asdict

import tyro

from config import TrainConfig, WandbConfig, WandbDisabled, WandbEnabled
from data import SupportedDatasets
from models.config import BaselineConfig, DeltapropConfig, XGBoostConfig


def prepare_dataset(dataset: SupportedDatasets, generate_features: bool):
    # Lazy import here to prevent cli startup from being slow
    import ray

    from data.loaders import load_dataset
    from data.preprocessing import preprocess_ray

    ray.init(ignore_reinit_error=True, num_cpus=4)

    df, df_classification_threshold = load_dataset(dataset)
    df = preprocess_ray(df, generate_features=generate_features)

    ray.shutdown()

    return df, df_classification_threshold


def wandb_log_artifacts(result_dict, predictions, split):
    import wandb
    artifact = wandb.Artifact( # type: ignore
        name=f"{result_dict['outer']}x{result_dict['inner']}_artifacts",
        type="generic"
    )
    
    with artifact.new_file("predictions.pkl", mode="wb") as f:
        pred_probs, preds = predictions
        pickle.dump({"pred_probs": pred_probs, "preds": preds}, f)

    with artifact.new_file("split.pkl", mode="wb") as f:
        train_df, val_df, test_df = split
        pickle.dump({"train": train_df, "cal": val_df, "test": test_df}, f)

    wandb.run.log_artifact(artifact) # type: ignore



def baseline(
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
    model_cf: BaselineConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
):
    from evaluate.train import train_and_evaluate
    from models.baseline import ChempropRef

    if isinstance(wandb_cf, WandbEnabled):
        import wandb

        wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
        tags = set(wandb_cf.tags) | set([
            'baseline', 
            dataset.name.lower(), 
            train_cf.split_type,
        ])

        if train_cf.use_feats:
            tags = tags | set(['with-feats'])
            
        run = wandb.init(project=wandb_cf.project_name, tags=list(tags))
        run.mark_preempting()

    df, df_classification_threshold = prepare_dataset(
        dataset, generate_features=train_cf.use_feats
    )
    result_iter = train_and_evaluate(
        df=df,
        df_classification_threshold=df_classification_threshold,
        model_class=ChempropRef,
        model_config=model_cf,
        train_config=train_cf,
    )

    for result_dict, predictions, split in result_iter:
        if isinstance(wandb_cf, WandbEnabled):
            model_name_suffix = wandb_cf.model_name_suffix
            model_name = "baseline" + (f"-{model_name_suffix}" if model_name_suffix else "")
            wandb.log(  # type: ignore
                result_dict
                | asdict(model_cf)
                | asdict(train_cf)
                | dict(dataset=dataset, model=model_name)
            )

            wandb_log_artifacts(result_dict, predictions, split)


        print(result_dict)

    return None


def deltaprop(
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
    model_cf: DeltapropConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
):
    from evaluate.train import train_and_evaluate
    from models.deltaprop import DeltapropRef

    if isinstance(wandb_cf, WandbEnabled):
        import wandb

        wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
        tags = set(wandb_cf.tags) | set([
            'deltaprop', 
            dataset.name.lower(), 
            train_cf.split_type,
        ])

        if train_cf.use_feats:
            tags = tags | set(['with-feats'])

        run = wandb.init(project=wandb_cf.project_name, tags=list(tags))
        run.mark_preempting()

    df, df_classification_threshold = prepare_dataset(
        dataset, generate_features=train_cf.use_feats
    )
    result_iter = train_and_evaluate(
        df=df,
        df_classification_threshold=df_classification_threshold,
        model_class=DeltapropRef,
        model_config=model_cf,
        train_config=train_cf,
    )

    for result_dict, predictions, split in result_iter:
        if isinstance(wandb_cf, WandbEnabled):
            model_name_suffix = wandb_cf.model_name_suffix
            model_name = "deltaprop" + (f"-{model_name_suffix}" if model_name_suffix else "")
            wandb.log(  # type: ignore
                result_dict
                | asdict(model_cf)
                | asdict(train_cf)
                | dict(dataset=dataset, model=model_name)
            )

            wandb_log_artifacts(result_dict, predictions, split)

        print(result_dict)

    return None


def xgboost(
    dataset: SupportedDatasets,
    train_cf: TrainConfig,
    model_cf: XGBoostConfig,
    wandb_cf: WandbConfig = WandbDisabled(),
):
    from evaluate.train import train_and_evaluate
    from models.xgboost_bl import XGBoostRef

    if isinstance(wandb_cf, WandbEnabled):
        import wandb

        wandb.login(key="cf344975eb80edf6f0d52af80528cc6094234caf")
        tags = set(wandb_cf.tags) | set([
            'xgboost', 
            dataset.name.lower(), 
            train_cf.split_type,
        ])

        if train_cf.use_feats:
            tags = tags | set(['with-feats'])

        run = wandb.init(project=wandb_cf.project_name, tags=list(tags))
        run.mark_preempting()

    df, df_classification_threshold = prepare_dataset(
        dataset, generate_features=train_cf.use_feats
    )
    result_iter = train_and_evaluate(
        df=df,
        df_classification_threshold=df_classification_threshold,
        model_class=XGBoostRef,
        model_config=model_cf,
        train_config=train_cf,
    )

    for result_dict, predictions, split in result_iter:
        if isinstance(wandb_cf, WandbEnabled):
            model_name_suffix = wandb_cf.model_name_suffix
            model_name = "xgboost" + (f"-{model_name_suffix}" if model_name_suffix else "")
            wandb.log(  # type: ignore
                result_dict
                | asdict(model_cf)
                | asdict(train_cf)
                | dict(dataset=dataset, model=model_name)
            )

            wandb_log_artifacts(result_dict, predictions, split)

        print(result_dict)

    return None



if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        dict(
            baseline=baseline, 
            deltaprop=deltaprop, 
            xgboost=xgboost
        )
    )
