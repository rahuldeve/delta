# Setup
Package management is taken care by `uv`. First install `uv` using any one of the specified steps here: https://docs.astral.sh/uv/getting-started/installation/

Next run `uv sync` inside the project folder. This will install all the necessary packages in a virtual environment inside the project folder named `.venv`

# Running experiments

`evaluate.cli` is the main entry point for running evaluation scripts. Use `uv` to run the evaluation script for a particular model and database combination. Here is a common to evaluate Chemprop the PK dataset using a random split approach
```bash
uv run --active python -m evaluate.cli \
chemprop \
--dataset PK \
--train-cf.split-type RANDOM
```
The first line uses `uv` to execute the `evaluate.cli` module. The rest are arguments that define the model, dataset, model and train configurations.

The first argument `chemprop` denotes the model to be used for evaluation. Currently there are 3 models supported: `chemprop`, `deltaprop` and `xgboost`

`--dataset PK` denotes the dataset to be used for evaluation

To view all available train and model configurations available for a specific model run the following command:
```bash
uv run --active python -m evaluate.cli chemprop --help
```

# Codebase Structure

`data/`: data loading and preprocessing functions
`datasets/`: raw dataset files
`evaluate`: evaluation script code
`experiments`: not relevant. Mostly just notebooks where I was trying ideas out
`models`: supported models code
