"""Запуск AutoGluon и FEDOT на одной и той же выборке.

Тяжёлые зависимости импортируются внутри функций, чтобы остальные модули
(метрики, графики, тесты) работали без установки AutoML-фреймворков.
"""

import time

import numpy as np

from .config import BenchmarkConfig
from .datasets import TARGET, Split
from .metrics import classification_metrics, regression_metrics


def _positive_class_proba(proba) -> np.ndarray:
    proba = np.asarray(proba)
    return proba[:, -1] if proba.ndim == 2 else proba


def run_autogluon(problem: str, split: Split, cfg: BenchmarkConfig) -> dict:
    from autogluon.tabular import TabularPredictor

    metric = "roc_auc" if problem == "classification" else "rmse"
    started = time.time()
    predictor = TabularPredictor(label=TARGET, eval_metric=metric, verbosity=0).fit(
        train_data=split.train_frame,
        time_limit=cfg.time_limit,
        presets=cfg.autogluon_preset,
    )
    elapsed = time.time() - started

    pred = predictor.predict(split.X_test)
    if problem == "classification":
        proba = predictor.predict_proba(split.X_test).iloc[:, 1]
        scores = classification_metrics(split.y_test, pred, proba)
    else:
        scores = regression_metrics(split.y_test, pred)
    return {**scores, "time": elapsed}


def run_fedot(problem: str, split: Split, cfg: BenchmarkConfig) -> dict:
    from fedot.api.main import Fedot

    is_clf = problem == "classification"
    model = Fedot(
        problem=problem,
        timeout=cfg.time_limit / 60,  # FEDOT принимает минуты
        seed=cfg.seed,
        metric="roc_auc" if is_clf else "rmse",
        cv_folds=cfg.fedot_cv_folds,
        with_tuning=True,
        logging_level=40,
        available_operations=list(cfg.fedot_classification_ops if is_clf else cfg.fedot_regression_ops),
    )
    started = time.time()
    model.fit(features=split.X_train.values, target=split.y_train.values)
    elapsed = time.time() - started

    pred = model.predict(features=split.X_test.values)
    if is_clf:
        proba = _positive_class_proba(model.predict_proba(features=split.X_test.values))
        scores = classification_metrics(split.y_test, pred, proba)
    else:
        scores = regression_metrics(split.y_test, pred)
    return {**scores, "time": elapsed}
