from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split

TARGET = "target"


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def train_frame(self) -> pd.DataFrame:
        """Обучающая выборка одной таблицей (формат AutoGluon)."""
        frame = self.X_train.copy()
        frame[TARGET] = self.y_train.to_numpy()
        return frame


def classification_split(test_size: float, seed: int) -> Split:
    """Breast Cancer Wisconsin: бинарная классификация, стратифицированный сплит."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name=TARGET)
    return Split(*train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y))


def regression_split(test_size: float, seed: int) -> Split:
    """California Housing: регрессия медианной стоимости жилья."""
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name=TARGET)
    return Split(*train_test_split(X, y, test_size=test_size, random_state=seed))
