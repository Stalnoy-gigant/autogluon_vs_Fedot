import json
from pathlib import Path

import numpy as np
import pytest

from automl_benchmark.datasets import TARGET, classification_split
from automl_benchmark.metrics import classification_metrics, regression_metrics, winner
from automl_benchmark.plots import plot_comparison

REFERENCE = Path(__file__).parent.parent / "results" / "reference_run" / "results.json"


def test_classification_split_is_stratified_and_reproducible():
    a = classification_split(test_size=0.2, seed=42)
    b = classification_split(test_size=0.2, seed=42)
    assert len(a.X_test) == pytest.approx(0.2 * (len(a.X_train) + len(a.X_test)), abs=1)
    assert a.y_test.mean() == pytest.approx(a.y_train.mean(), abs=0.02)
    assert a.X_test.index.equals(b.X_test.index)
    assert TARGET in a.train_frame.columns


def test_classification_metrics_perfect_prediction():
    y = np.array([0, 1, 1, 0])
    scores = classification_metrics(y, y, y.astype(float))
    assert scores == {"accuracy": 1.0, "f1": 1.0, "roc_auc": 1.0}


def test_regression_metrics_known_values():
    scores = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
    assert scores["mae"] == pytest.approx(2 / 3)
    assert scores["rmse"] == pytest.approx(np.sqrt(2 / 3))


@pytest.mark.parametrize(
    "metric, ag, fedot, expected",
    [("roc_auc", 0.99, 0.98, "autogluon"), ("rmse", 0.7, 0.4, "fedot"),
     ("f1", 0.9, 0.9, "tie"), ("mae", 0.2, 0.5, "autogluon")],
)
def test_winner_respects_metric_direction(metric, ag, fedot, expected):
    assert winner(metric, ag, fedot) == expected


def test_reference_results_render(tmp_path):
    payload = json.loads(REFERENCE.read_text())
    out = tmp_path / "plot.png"
    plot_comparison(payload["results"], payload["time_limit"], out)
    assert out.stat().st_size > 10_000
