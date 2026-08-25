# models/__init__.py
"""
Model factory: returns RegressorBaseline or Regressor based on model_type.
Exports Regressor, RegressorBaseline, CrossAttention.
"""
from .main_model import Regressor
from .no_attention import RegressorBaseline
from .attention import CrossAttention

__all__ = [
    'Regressor',
    'RegressorBaseline',
    'CrossAttention',
    'create_model'
]


def create_model(model_type=None, config=None):
    if model_type == 'baseline':
        return RegressorBaseline(
            feature_dim=config['model']['feature_dim'],
            dropout=config['model']['dropout'],
            hidden_dims=config['model']['hidden_dims'],
            gene_input_dim=config['data']['gene_input_dim'],
            drug_global_dim=config['data']['drug_global_dim'],
            config=config
        )
    elif model_type == 'attention':
        return Regressor(
            feature_dim=config['model']['feature_dim'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout'],
            hidden_dims=config['model']['hidden_dims'],
            gene_input_dim=config['data']['gene_input_dim'],
            drug_global_dim=config['data']['drug_global_dim'],
            num_fusion_layers=config['model']['num_fusion_layers'],
            config=config
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
