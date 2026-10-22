import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

# Для каждой метрики: True, если меньше значит лучше
LOWER_IS_BETTER = {"mae": True, "rmse": True, "r2": False,
                   "accuracy": False, "f1": False, "roc_auc": False}


def classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def winner(metric: str, autogluon_value: float, fedot_value: float) -> str:
    """Какой фреймворк лучше по метрике; при равенстве возвращает 'tie'."""
    if autogluon_value == fedot_value:
        return "tie"
    ag_better = (autogluon_value < fedot_value) == LOWER_IS_BETTER[metric]
    return "autogluon" if ag_better else "fedot"
