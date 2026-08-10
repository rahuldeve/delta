from data import SupportedDatasets


def prepare_dataset(
    dataset: SupportedDatasets, use_features: bool, drop_nan_features: bool
):
    # Lazy import here to prevent cli startup from being slow
    import ray

    from data.loaders import load_dataset
    from data.preprocessing import preprocess_ray

    ray.init(ignore_reinit_error=True, num_cpus=4)

    df, df_classification_threshold = load_dataset(dataset)
    df = preprocess_ray(
        df, use_features=use_features, drop_nan_features=drop_nan_features
    )

    ray.shutdown()

    return df, df_classification_threshold
