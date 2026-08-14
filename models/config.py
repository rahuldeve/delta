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
    encoder_hidden_dim: int = 600
    encoder_output_dim: int = 300
    encoder_n_layers: int = 4
    encoder_dropout: float = 0.1
    batch_norm: bool = False
    interaction_dropout: float = 0.1
    # Decoupled (AdamW) weight decay, applied to weight matrices only — biases,
    # LayerNorm/BatchNorm gains and the Davidson tie parameter are exempt.
    # 0.0 makes the optimizer identical to plain Adam.
    weight_decay: float = 0.01
    candidate_size: int = 24
    frac_hard: float = 0.5
    # Half-width of the band around the decision boundary that counts as hard.
    # Mining peaks on pairs the model is undecided about and falls to zero at
    # |agreement| = hard_band; smaller values keep only the truly undecided.
    hard_band: float = 1.0
    use_chameleon_mp: bool = False


@dataclass
class XGBoostConfig(ModelConfig):
    random_state: int = 42
