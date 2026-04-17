"""Training and evaluation utilities for the Effort Evaluator."""

from eml_transformer.training.baselines import GlobalMeanBaseline, TokenClassBaseline
from eml_transformer.training.metrics import DepthMetrics, compute_metrics, pretty_print_metrics
from eml_transformer.training.trainer import TrainConfig, evaluate, train

__all__ = [
    "GlobalMeanBaseline",
    "TokenClassBaseline",
    "DepthMetrics",
    "compute_metrics",
    "pretty_print_metrics",
    "TrainConfig",
    "evaluate",
    "train",
]
