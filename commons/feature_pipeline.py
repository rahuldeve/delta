from feature_generation import MorganFP
from feature_selection import CorrelationThreshold
from lightgbm import LGBMRegressor
from ray import tune
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FunctionTransformer, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler


def get_pipeline(random_seed):
    feature_filters = make_pipeline(
        FunctionTransformer(
            lambda df: df.drop(["inchi", "smiles", "mol"], axis=1, errors="ignore")
        ),
        CorrelationThreshold(0.95),
        VarianceThreshold(0.1),
        StandardScaler(),
    )

    pipeline = make_pipeline(
        feature_filters,
        LGBMRegressor(random_state=random_seed),
    )

    return pipeline


def get_pipeline_param_space(random_seed):
    pipeline_param_space = {
        "pipeline__variancethreshold__threshold": tune.uniform(0.05, 0.3),
        "pipeline__correlationthreshold__threshold": tune.uniform(
            0.8, 1.0
        ),
        "lgbmregressor__objective": "binary",
        "lgbmregressor__metric": "average_precision",
        "lgbmregressor__verbosity": -1,
        "lgbmregressor__boosting_type": "dart",
        "lgbmregressor__reg_alpha": tune.loguniform(1e-8, 1e-1),
        "lgbmregressor__reg_lambda": tune.loguniform(1e-8, 1e-1),
        "lgbmregressor__num_leaves": tune.randint(2, 256),
        "lgbmregressor__subsample": tune.uniform(0.1, 1),
        "lgbmregressor__colsample_bytree": tune.uniform(0.1, 1),
        "lgbmregressor__min_child_samples": tune.randint(5, 100),
        "lgbmregressor__n_jobs": 4,
        "lgbmregressor__random_state": random_seed,
        "lgbmregressor__scale_pos_weight": tune.qrandint(30, 100, 2),
        "lgbmregressor__n_estimators": tune.randint(50, 1000),
        "lgbmregressor__max_depth": tune.randint(5, 50),
        # "lgbmregressor__early_stopping_rounds": ConstantParam(100)
    }

    return pipeline_param_space
