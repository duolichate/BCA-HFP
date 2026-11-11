# bca_hfp/config/base_config.py
import copy
import os

PROJECT_ROOT = os.environ.get('BCA_HFP_ROOT') or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# subdirectories may carry a numeric prefix (1_Data, 5_Result); both layouts resolve
_PREFIX = {"Data": "1_", "Drug_Feature": "2_", "Gene_Feature": "3_",
           "Result": "5_", "Picture": "6_"}


def root_dir(name):
    """Resolve name to its actual path under the project root."""
    prefix = _PREFIX.get(name)
    if prefix:
        cand = os.path.join(PROJECT_ROOT, prefix + name)
        if os.path.isdir(cand):
            return cand
    return os.path.join(PROJECT_ROOT, name)


_ENV_ALIASES = {'loc': '7900XT', 'srv': 'TeslaT4', 'cpu': 'cpu'}


def detect_env(choice='auto'):
    """Return a machine tag derived from the GPU name."""
    if choice and choice != 'auto':
        return _ENV_ALIASES.get(choice, choice)
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0).upper()
            if '5090' in dev:
                return 'RTX5090'
            if 'T4' in dev:
                return 'TeslaT4'
            if '7900' in dev or 'RADEON' in dev or 'AMD' in dev:
                return '7900XT'
            return ''.join(ch for ch in dev.title() if ch.isalnum())[:16] or 'gpu'
    except Exception:
        pass
    return 'cpu'


def get_base_config(data):
    base_data_path = root_dir("Data")
    base_result_path = root_dir("Result")

    if data == 'GDSC':
        return {
            "data": {
                # Data column names
                "drug_id_col": "DRUG_ID",  # Drug ID
                "cell_id_col": "GDSC_ID",  # Cell line ID
                "label_col": "LN_IC50",  # Prediction label column
                # Feature file paths (under the Drug/ and Gene/ subdirectories)
                "drug_feat_path": f"{base_data_path}/Drug/GDSC_RES_FGBERT.h5",
                "gene_feat_dir": f"{base_data_path}/Gene/GDSC_EXP_GF_V1/",
                "response_path": f"{base_data_path}/Drug/GDSC_RES_FGBERT.h5",
                # Random seed
                "random_state": 42
            },
            "training": {
                "batch_size": 32,  # Batch size
                "gradient_accumulation_steps": 4,  # Gradient accumulation steps
                "epochs": 200,  # Number of epochs
                "lr": 2e-5,  # Learning rate
                "weight_decay": 1e-3,  # Weight decay
                "gradient_clip": 0.5,  # Gradient clipping
                "early_stop_patience": 10,  # Early stopping patience
                "num_workers": 8,
                "atom_mask_prob": 0.0,
                "scheduler": "reduce_on_plateau",  # Learning rate scheduler type
                "reduce_factor": 0.5,
                "reduce_patience": 5,
                "drug_global_dropout": 0.3,  # Dropout probability for drug global features
                "global_gene_dropout": 0.3,  # Dropout probability for gene global features
                "gate_temperature": 1.0,
            },
            "model": {
                "feature_dim": 256,
                "num_heads": 8,
                "dropout": 0.2,
                "model_name": "attention",
                "hidden_dims": [512, 256, 128],
                "num_fusion_layers": 1
            },
            "output": {
                "model_save_dir": f"{base_result_path}/{data}/models",  # Model save directory
                "attention_csv_dir": f"{base_result_path}/{data}/attentions",  # Attention weights CSV directory
                "result_save_dir": f"{base_result_path}/{data}/results"  # Result save directory
            }
        }
    else:
        raise ValueError(f"Unknown dataset: {data}")


def update_config(base_config, updates):
    config = copy.deepcopy(base_config)
    for key, value in updates.items():
        if isinstance(value, dict) and key in config:
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config
