from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkConfig:
    """Параметры эксперимента: одинаковые для обоих фреймворков."""

    time_limit: int = 120  # секунд на одну задачу и один фреймворк
    seed: int = 42
    test_size: float = 0.2
    fedot_cv_folds: int = 3
    # Ограниченный набор моделей FEDOT: сопоставимый по составу с AutoGluon medium_quality
    fedot_classification_ops: tuple[str, ...] = ("logit", "rf", "knn", "dt", "mlp")
    fedot_regression_ops: tuple[str, ...] = ("linear", "ridge", "lasso", "rf", "dt", "knn")
    autogluon_preset: str = "medium_quality"
