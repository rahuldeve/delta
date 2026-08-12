from dataclasses import dataclass

import tyro

from models.abc import ModelConfig

Group = tyro.conf.create_mutex_group(required=False, title="something")


@dataclass
class ChempropConfig(ModelConfig):
    mp_d_h: int = 300
    mp_depth: int = 3
    mp_dropout: float = 0.0
    ffn_hidden_dim: int = 300
    ffn_n_layers: int = 2
    ffn_dropout: float = 0.1
    batch_norm: bool = False
    use_chameleon_mp: bool = False


@dataclass
class DeltapropConfig(ModelConfig):
    mp_d_h: int = 300
    mp_depth: int = 3
    mp_dropout: float = 0.0
    encoder_hidden_dim: int = 300
    encoder_output_dim: int = 300
    encoder_n_layers: int = 2
    encoder_dropout: float = 0.1
    batch_norm: bool = False
    interaction_dropout: float = 0.0
    candidate_size: int = 32
    frac_hard: float = 0.2
    # How much of a correct-ordering lead still counts as a hard pair. 0.0 mines
    # only pairs the model gets outright wrong; >0 also pulls in near misses.
    hard_margin: float = 0.3
    use_chameleon_mp: bool = False


@dataclass
class XGBoostConfig(ModelConfig):
    random_state: int = 42
