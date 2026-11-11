# bca_hfp/utils/reproducibility.py
import torch
import numpy as np
import random
import os


def set_seed(seed=42, deterministic=True):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False, warn_only=True)
    print(f"Random seed set to {seed} (deterministic={deterministic})")
