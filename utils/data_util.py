"""Data Utils Functions.

Utility functions for loading datasets and preprocessing (train-test split, scaling).
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(
    dataset_name: Literal["california", "diabetes"],
    features: Optional[list[str]] = None,
    as_numpy: bool = True,
) -> Tuple[
    Union[pd.DataFrame, NDArray[np.float64]], Union[pd.Series, NDArray[np.float64]]
]:
    """Load a dataset and return selected features and target.

    Args:
        dataset_name: Name of the dataset ('california', 'diabetes').
        features: List of features to select (optional, selects all by default).
        as_numpy: If True, return numpy arrays; else pandas DataFrame/Series.

    Returns:
        Tuple of features (X) and target (y).
    """
    if dataset_name == "california":
        dataset = fetch_california_housing()
        X_full = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y_full = pd.Series(dataset.target)
    elif dataset_name == "diabetes":
        dataset = load_diabetes()
        X_full = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y_full = pd.Series(dataset.target)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if features:
        X_full = X_full[features]

    if as_numpy:
        return X_full.to_numpy(dtype=np.float64), y_full.to_numpy(dtype=np.float64)

    return X_full, y_full


def train_test_split_scaled(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Perform train-test split and scale features.

    Args:
        x: Feature matrix.
        y: Target values.
        test_size: Test set size fraction.
        random_state: Random seed.

    Returns:
        Tuple of scaled train-test features and targets.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train, y_test
