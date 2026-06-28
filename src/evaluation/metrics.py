"""
Computes all evaluation metrics from a RunResult.
Returns a flat dict ready for pandas DataFrame construction.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)
from src.models.base import RunResult  # noqa — import from base


def compute_metrics(result: RunResult) -> dict:
    """
    Given a RunResult, compute all metrics and return as a flat dict.
    Returns NaN for metrics that cannot be computed (e.g., error run).
    """
    base = {
        "model": result.model_name,
        "dataset": result.dataset_name,
        "seed": result.seed,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_features": result.n_features,
        "fit_time_sec": result.fit_time_sec,
        "predict_time_sec": result.predict_time_sec,
        "total_time_sec": result.fit_time_sec + result.predict_time_sec,
        "tuning_time_sec": result.tuning_time_sec,
        "total_wall_time_sec": result.total_wall_time_sec,
        "throughput_rows_per_sec": result.throughput_rows_per_sec,
        "ms_per_test_row": (result.predict_time_sec / result.n_test) * 1000 if result.n_test > 0 else None,
        "error": result.error,
    }

    if result.error is not None:
        # Fill all metrics with NaN for failed runs
        base.update({k: np.nan for k in [
            "roc_auc", "avg_precision", "f1_macro",
            "brier_score", "log_loss_val",
        ]})
        return base

    y = result.y_test
    yp = result.y_proba
    yh = result.y_pred

    # Clip probabilities to avoid log(0)
    yp_clipped = np.clip(yp, 1e-7, 1 - 1e-7)

    base.update({
        "roc_auc":       roc_auc_score(y, yp),
        "avg_precision": average_precision_score(y, yp),
        "f1_macro":      f1_score(y, yh, average="macro", zero_division=0),
        "brier_score":   brier_score_loss(y, yp),
        "log_loss_val":  log_loss(y, yp_clipped),
    })

    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    base.update({
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,  # recall
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "ppv":         tp / (tp + fp) if (tp + fp) > 0 else np.nan,  # precision
    })

    return base