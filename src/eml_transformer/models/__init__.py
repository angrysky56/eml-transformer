"""Model package for the effort-gated transformer."""

from eml_transformer.models.decoder import (
    VALID_FFN_MODES,
    ModelConfig,
    TinyDecoder,
    make_config,
)
from eml_transformer.models.effort_head import EffortHead
from eml_transformer.models.layers import (
    CausalSelfAttention,
    DecoderLayer,
    DeltaFFN,
    FeedForward,
    FiLMFFN,
    LMHead,
    RotaryEmbedding,
    SelfAwareFFN,  # backward-compat alias for DeltaFFN
    make_ffn,
)
from eml_transformer.models.self_aware import EMLTransformer

__all__ = [
    "ModelConfig",
    "TinyDecoder",
    "EffortHead",
    "CausalSelfAttention",
    "DecoderLayer",
    "FeedForward",
    "DeltaFFN",
    "FiLMFFN",
    "SelfAwareFFN",
    "LMHead",
    "RotaryEmbedding",
    "EMLTransformer",
    "VALID_FFN_MODES",
    "make_config",
    "make_ffn",
]
