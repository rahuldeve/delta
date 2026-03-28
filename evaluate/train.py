from typing import Type

import numpy as np
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
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from config import SplitType, TrainConfig
from data import DSThreshold
from models.abc import RefModel
from models.config import ModelConfig


def get_group_splitters(random_state, n_outer):
    outer_splitter = StratifiedGroupKFold(
        n_splits=n_outer,
        shuffle=True,  # type: ignore
        random_state=random_state,  # type: ignore
    )
    # Since StratifiedGroupShuffleSplit does not exist, we can use GroupShuffleSplit for
    # splitting val and test to get around this issue
    # ref: https://github.com/scikit-learn/scikit-learn/issues/12076#issuecomment-2047948563
    inner_spliter = StratifiedKFold(
        n_splits=int(1 / 0.5), shuffle=True, random_state=random_state
    )
    return outer_splitter, inner_spliter


def get_random_splitters(random_state, n_outer):
    outer_splitter = StratifiedKFold(
        n_splits=n_outer, shuffle=True, random_state=random_state
    )
    inner_spliter = StratifiedShuffleSplit(1, test_size=0.5, random_state=random_state)
    return outer_splitter, inner_spliter


def generate_repeated_5xn_splits(df, n: int, split_type: SplitType, random_state: int):
    rng = np.random.RandomState(random_state)
    for outer_idx in range(5):
        randint = rng.randint(low=0, high=32767)

        if split_type == SplitType.RANDOM:
            outer_splitter, inner_spliter = get_random_splitters(randint, n_outer=n)
            group_col_getter = lambda _df: None  # noqa: E731
        elif split_type == SplitType.SCAFFOLD:
            outer_splitter, inner_spliter = get_group_splitters(randint, n_outer=n)
            group_col_getter = lambda _df: _df["scaffold_cluster"]  # noqa: E731
        elif split_type == SplitType.BUTINA:
            outer_splitter, inner_spliter = get_group_splitters(randint, n_outer=n)
            group_col_getter = lambda _df: _df["butina_cluster"]  # noqa: E731
        else:
            raise ValueError(split_type)

        for inner_idx, (train_idxs, val_test_idxs) in enumerate(
            outer_splitter.split(df, y=df["bin_target"], groups=group_col_getter(df))
        ):
            train_df = df.loc[train_idxs].reset_index(drop=True)
            val_test_df = df.loc[val_test_idxs].reset_index(drop=True)

            val_idxs, test_idxs = next(
                inner_spliter.split(
                    val_test_df,
                    y=val_test_df["bin_target"],
                    groups=group_col_getter(val_test_df),
                )
            )

            val_df = val_test_df.loc[val_idxs].reset_index(drop=True)
            test_df = val_test_df.loc[test_idxs].reset_index(drop=True)

            yield (outer_idx, inner_idx), (train_df, val_df, test_df)


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
    model_class: Type[RefModel],
    model_config: ModelConfig,
    train_config: TrainConfig,
):
    train_split, val_split, test_split, extras = model_class.prepare_splits(
        train_df=train_df, val_df=val_df, test_df=test_df
    )

    model = model_class.build(model_config=model_config, **extras)

    model = model.train_func(
        train_split=train_split,
        val_split=val_split,
        train_config=train_config,
        model_config=model_config,
        df_classification_threshold=df_classification_threshold,
    )

    clf_th = model.tune_binary_classification_threshold(
        val_split=val_split,
        val_labels=val_df["bin_target"],
        train_config=train_config,
        train_split=train_split,
        train_labels=train_df["bin_target"],
        df_classification_threshold=df_classification_threshold,
    )

    pred_probs, preds = model.predict_func(
        binary_classification_threshold=clf_th,
        train_split=train_split,
        train_labels=train_df["bin_target"],
        test_split=test_split,
        df_classification_threshold=df_classification_threshold,
    )

    return calc_metrics(pred_probs, preds, test_df["bin_target"]), (pred_probs, preds)


def train_and_evaluate(
    df,
    df_classification_threshold: DSThreshold,
    model_class: Type[RefModel],
    model_config: ModelConfig,
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

        metrics_dict, predictions = train_and_evaluate_split(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            df_classification_threshold=df_classification_threshold,
            model_class=model_class,
            model_config=model_config,
            train_config=train_config,
        )

        result_dict = {"outer": outer_idx, "inner": inner_idx} | metrics_dict
        yield result_dict, predictions, (train_df, val_df, test_df)
