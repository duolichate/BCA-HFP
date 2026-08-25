"""
Configuration module: centralized hyperparameter management with experiment versioning support.
"""
from .base_config import get_base_config, update_config

__all__ = [
    'get_base_config',
    'update_config'
]