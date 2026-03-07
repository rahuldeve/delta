from dataclasses import dataclass

import tyro

Group = tyro.conf.create_mutex_group(required=False, title="something")


@dataclass
class BaselineConfig:
    mp_d_h: int = 300
    mp_depth: int = 3
    mp_dropout: float = 0.0
    ffn_hidden_dim: int = 300
    ffn_n_layers: int = 2
    ffn_dropout: float = 0.1
    batch_norm: bool = False
    use_chameleon_mp: bool = False


@dataclass
class DeltapropConfig:
    mp_d_h: int = 300
    mp_depth: int = 3
    mp_dropout: float = 0.0
    encoder_hidden_dim: int = 300
    encoder_output_dim: int = 300
    encoder_n_layers: int = 2
    encoder_dropout: float = 0.1
    batch_norm: bool = False
    interaction_dropout: float = 0.0
    candidate_size: int = 16
    use_chameleon_mp: bool = False
