"""Linear Regression Usage Example Module."""

from typing import Literal, Optional

import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from algorithms.linear_regression import LinearRegression
from utils.data_util import load_dataset, train_test_split_scaled


def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> None:
    """Print evaluation metrics for a given model."""
    print(f"\n Evaluation Metrics - {model_name}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"MSE     : {mean_squared_error(y_true, y_pred):.4f}")
    print(f"MAE     : {mean_absolute_error(y_true, y_pred):.4f}")


def print_coefficients(custom_weights: np.ndarray, sk_weights: np.ndarray) -> None:
    """Print model coefficients side by side."""
    print("\nModel Coefficients:")
    print(f"Custom LinearRegression Weights:     {np.round(custom_weights, 4)}")
    print(f"Scikit-Learn LinearRegression Coeff.: {np.round(sk_weights, 4)}")


def run_pipeline(
    dataset: Literal["california", "diabetes"] = "california",
    features: Optional[list[str]] = None,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
) -> None:
    """Run an end-to-end ML pipeline.

    Compares custom and sklearn Linear Regression side-by-side.
    """
    if features is None:
        features = ["MedInc", "AveRooms", "AveOccup"]
    print(f"\nRunning pipeline on dataset: {dataset} with features: {features}")

    # Load and split data
    x, y = load_dataset(dataset_name=dataset, features=features, as_numpy=True)
    x_train, x_test, y_train, y_test = train_test_split_scaled(x, y)

    # Custom Linear Regression
    custom_model = LinearRegression(
        learning_rate=learning_rate, n_iterations=n_iterations
    )
    custom_model.fit(x_train, y_train)
    y_pred_custom = custom_model.predict(x_test)

    # Scikit-learn Linear Regression
    sk_model = SklearnLinearRegression()
    sk_model.fit(x_train, y_train)
    y_pred_sk = sk_model.predict(x_test)

    # Evaluation
    evaluate_model(y_test, y_pred_custom, model_name="Custom LinearRegression")
    evaluate_model(y_test, y_pred_sk, model_name="Scikit-Learn LinearRegression")

    # Custom Huber Loss
    huber_loss = custom_model.score(x_test, y_test, metric="huber", delta=1.0)
    print(f"Huber Loss (Custom Model): {huber_loss:.4f}")

    # Compare Coefficients
    print_coefficients(custom_model.weights, sk_model.coef_)


if __name__ == "__main__":
    run_pipeline()
