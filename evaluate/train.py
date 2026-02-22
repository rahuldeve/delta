import chemprop as cp
import numpy as np
from evaluate.config import SplitType, TrainConfig
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit

from data import DSThreshold
from models.config import BaselineConfig, DeltapropConfig


def get_group_splitters(random_state, n_outer):
    outer_splitter = GroupKFold(
        n_splits=n_outer,
        shuffle=True,  # type: ignore
        random_state=random_state,  # type: ignore
    )
    inner_spliter = GroupShuffleSplit(1, test_size=0.5, random_state=random_state)
    return outer_splitter, inner_spliter


def get_random_splitters(random_state, n_outer):
    outer_splitter = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    inner_spliter = ShuffleSplit(1, test_size=0.5, random_state=random_state)
    return outer_splitter, inner_spliter


def generate_repeated_5xn_splits(df, n: int, split_type: SplitType, random_state: int):
    rng = np.random.RandomState(random_state)
    for outer_idx in range(5):
        randint = rng.randint(low=0, high=32767)

        if split_type == SplitType.RANDOM:
            outer_splitter, inner_spliter = get_random_splitters(randint, n_outer=n)
        elif split_type == SplitType.SCAFFOLD:
            outer_splitter, inner_spliter = get_group_splitters(randint, n_outer=n)
        else:
            raise ValueError(split_type)

        for inner_idx, (train_idxs, val_test_idxs) in enumerate(
            outer_splitter.split(df, groups=df["cluster"])
        ):
            train_df = df.loc[train_idxs].reset_index(drop=True)
            val_test_df = df.loc[val_test_idxs].reset_index(drop=True)

            val_idxs, test_idxs = next(
                inner_spliter.split(val_test_df, groups=val_test_df["cluster"])
            )

            val_df = val_test_df.loc[val_idxs].reset_index(drop=True)
            test_df = val_test_df.loc[test_idxs].reset_index(drop=True)

            yield (outer_idx, inner_idx), (train_df, val_df, test_df)


def prepare_mol_datasets(train_df, val_df, test_df, model_module):
    train_df["mol_dp"] = train_df.apply(model_module.get_molecule_datapoint, axis=1)
    val_df["mol_dp"] = val_df.apply(model_module.get_molecule_datapoint, axis=1)
    test_df["mol_dp"] = test_df.apply(model_module.get_molecule_datapoint, axis=1)

    featurizer = cp.featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_mol_dataset = cp.data.MoleculeDataset(
        train_df["mol_dp"], featurizer=featurizer
    )
    val_mol_dataset = cp.data.MoleculeDataset(val_df["mol_dp"], featurizer=featurizer)
    test_mol_dataset = cp.data.MoleculeDataset(test_df["mol_dp"], featurizer=featurizer)

    x_d_scaler = train_mol_dataset.normalize_inputs("X_d")
    val_mol_dataset.normalize_inputs("X_d", x_d_scaler)

    train_mol_dataset.cache = True
    val_mol_dataset.cache = True

    return train_mol_dataset, val_mol_dataset, test_mol_dataset, x_d_scaler


def calc_metrics(pred_probs, preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "mcc": matthews_corrcoef(labels, preds),
        "roc_auc": roc_auc_score(labels, pred_probs),
        "average_precision": average_precision_score(labels, pred_probs),
    }


def train_and_evaluate_split(
    train_df,
    val_df,
    test_df,
    df_classification_threshold: DSThreshold,
    model_module,
    model_config: DeltapropConfig | BaselineConfig,
    train_config: TrainConfig,
):
    train_mol_ds, val_mol_ds, test_mol_ds, X_d_scaler = prepare_mol_datasets(
        train_df, val_df, test_df, model_module
    )

    model = model_module.train_func(
        config=model_config,
        train_mol_ds=train_mol_ds,
        val_mol_ds=val_mol_ds,
        X_d_scaler=X_d_scaler,
        binary_threshold=df_classification_threshold,
        batch_size=train_config.batch_size,
        max_epochs=train_config.max_epochs,
        early_stopping_patience=train_config.early_stopping_patience,
        random_seed=train_config.random_seed,
    )

    clf_th = model_module.tune_binary_classification_threshold(
        model=model,
        train_mol_ds=train_mol_ds,
        train_labels=train_df["bin_target"],
        val_mol_ds=val_mol_ds,
        val_labels=val_df["bin_target"],
        random_seed=train_config.random_seed,
        df_classification_threshold=df_classification_threshold
    )

    pred_probs, preds = model_module.predict_func(
        model=model,
        binary_classification_threshold=clf_th,
        train_mol_ds=train_mol_ds,
        train_labels=train_df["bin_target"],
        test_mol_ds=test_mol_ds,
        df_classification_threshold=df_classification_threshold
    )

    return calc_metrics(pred_probs, preds, test_df["bin_target"])


def train_and_evaluate(
    df,
    df_classification_threshold: DSThreshold,
    model_module,
    model_config: DeltapropConfig | BaselineConfig,
    train_config: TrainConfig,
):
    splits = generate_repeated_5xn_splits(
        df,
        train_config.n_splits,
        train_config.split_type,
        random_state=train_config.random_seed,
    )
    for split_idxs, split in splits:
        outer_idx, inner_idx = split_idxs
        train_df, val_df, test_df = split

        metrics_dict = train_and_evaluate_split(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            df_classification_threshold=df_classification_threshold,
            model_module=model_module,
            model_config=model_config,
            train_config=train_config,
        )

        result_dict = {"outer": outer_idx, "inner": inner_idx} | metrics_dict
        yield result_dict
