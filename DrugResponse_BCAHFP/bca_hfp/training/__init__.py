# bca_hfp/training/__init__.py

from .trainer import BaseTrainer, VariableLengthTrainer, FixedLengthTrainer
from .metrics import compute_metrics


Trainer = VariableLengthTrainer

__all__ = [
    'BaseTrainer',
    'VariableLengthTrainer',
    'FixedLengthTrainer',
    'Trainer',
    'compute_metrics'
]

