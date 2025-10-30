from chemprop.data import MoleculeDataset
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from sklearn.preprocessing import StandardScaler

from utils import RANDOM_SEED


def tune_model(
    tune_func,
    search_space,
    train_mol_ds: MoleculeDataset,
    val_mol_ds: MoleculeDataset,
    X_d_scaler: StandardScaler,
    batch_size: int,
    max_epochs: int,
    early_stopping_patience: int = 10,
    num_samples: int = 20,
    max_concurrent: int = 2,
    per_trial_gpu: float = 0.5,
    **extra_tune_fn_parameters,
):
    search_alg = ConcurrencyLimiter(
        OptunaSearch(seed=RANDOM_SEED), max_concurrent=max_concurrent
    )
    scheduler = ASHAScheduler(max_t=max_epochs, grace_period=5, reduction_factor=2)

    tune_fn = tune.with_resources(
        tune.with_parameters(
            tune_func,
            train_mol_ds=train_mol_ds,
            val_mol_ds=val_mol_ds,
            X_d_scaler=X_d_scaler,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            **extra_tune_fn_parameters,
        ),
        resources={"GPU": per_trial_gpu},
    )

    tuner = tune.Tuner(
        tune_fn,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=tune.RunConfig(
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
                checkpoint_at_end=False,
            ),
            storage_path="/tmp/ray_results",
            failure_config=tune.FailureConfig(max_failures=3),
        ),
    )

    results = tuner.fit()
    _, best_result = results.get_best_result("val_loss", "min").best_checkpoints[0]  # type: ignore
    return best_result["config"]
