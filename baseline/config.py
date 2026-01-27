from dataclasses import dataclass


@dataclass
class BaselineConfig:
    depth: int = 1
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 1
    message_hidden_dim: int = 300
    batch_norm: bool = False
    encoder_dropout: float = 0.0
