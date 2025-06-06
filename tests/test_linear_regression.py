"""This module contains utility functions for loading datasets."""

from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from algorithms.linear_regression import LinearRegression


@pytest.fixture
def sample_data() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fixture to generate simple linear data: y = 2x + 3."""
    X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    y = np.array([5, 7, 9, 11, 13], dtype=np.float64)
    return X, y


def test_fit_predict_r2(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Test R2 score calculation."""
    X, y = sample_data
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    r2 = model.score(X, y, metric="r2")
    assert r2 > 0.99


def test_predict_before_fit_raises() -> None:
    """Test Prediction flase use case."""
    model = LinearRegression()
    X = np.array([[1], [2], [3]], dtype=np.float64)
    with pytest.raises(ValueError):
        model.predict(X)


def test_score_metrics(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Testing Metrics of the model."""
    X, y = sample_data
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    assert model.score(X, y, metric="r2") > 0.99
    assert model.score(X, y, metric="rmse") < 0.7
    assert model.score(X, y, metric="mse") < 0.5
    assert model.score(X, y, metric="mae") < 0.5
    assert model.score(X, y, metric="huber", delta=1.0) < 0.5


def test_invalid_metric_raises(
    sample_data: Tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Invalid Use case of testing metrics."""
    X, y = sample_data
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.score(X, y, metric="unsupported_metric")

    with pytest.raises(TypeError):
        model.score(X, y, metric=12345)  # Invalid Type
