# bca_hfp/training/metrics.py
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np


def compute_metrics(y_true, y_pred, threshold=None):
    """Compute R², MAE, RMSE, Pearson correlation. Returns dict."""
    if threshold is not None:
        mask = y_true > threshold
        if np.sum(mask) == 0:
            raise ValueError(f"No valid samples after filtering with threshold {threshold}")
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0

    return {
        'r2': round(r2, 4),
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'pearson': round(pearson, 4)
    }
