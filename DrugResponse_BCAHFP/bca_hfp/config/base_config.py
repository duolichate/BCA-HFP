# config/base_config.py
import copy
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_base_config(data):
    base_data_path = os.path.join(PROJECT_ROOT, "Data")  # Data directory
    base_result_path = os.path.join(PROJECT_ROOT, "Result")  # Result directory

    if data == 'GDSC':
        return {
            "data": {
                # Data column names
                "drug_id_col": "DRUG_ID",  # Drug ID
                "cell_id_col": "GDSC_ID",  # Cell line ID
                "label_col": "LN_IC50",  # Prediction label column
                # Feature file paths
                "drug_feat_path": f"{base_data_path}/GDSC_RES_FGBERT.h5",
                "gene_feat_dir": f"{base_data_path}/GDSC_EXP_GF/",
                "response_path": f"{base_data_path}/GDSC_RES_FGBERT.h5",
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
                "gate_temperature": 1.5,  # >1 for more uniform weight distribution
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
