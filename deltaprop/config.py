from dataclasses import dataclass


@dataclass
class DeltapropConfig:
    depth: int = 1
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 1
    message_hidden_dim: int = 300
    batch_norm: bool = False
    encoder_dropout: float = 0.0
    interaction_dropout: float = 0.0
    candidate_size: int = 32


DEFAULT_CONFIG = DeltapropConfig(
    depth=1,
    ffn_hidden_dim=300,
    ffn_num_layers=1,
    message_hidden_dim=300,
    batch_norm=False,
    encoder_dropout=0.0,
    interaction_dropout=0.0,
    candidate_size=32,
)
