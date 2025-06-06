"""Linear Regression implementation from scratch using NumPy.

This module provides a LinearRegression class that supports fitting a
linear model to data using gradient descent optimization. It includes
methods for prediction, scoring (R-squared), and is designed for basic
regression analysis tasks.
"""

from typing import Any, Callable, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LinearRegression:
    """Linear Regression model using gradient descent.

    Attributes:
        learning_rate (float): Step size for gradient descent updates.
        n_iterations (int): Number of iterations for gradient descent.
        weights (Optional[NDArray[np.float64]]): Coefficients of the model.
        bias (Optional[float]): Intercept term of the model.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        """Initialize the LinearRegression model.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            n_iterations (int): The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: NDArray[np.float64] = np.array([])
        self.bias: float = 0.0

    @staticmethod
    def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Compute mean squared error (MSE) loss."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Compute root squared error (R2) loss."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return float(1 - (ss_residual / ss_total))

    @staticmethod
    def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Compute root mean squared error (RMSE) loss."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(np.square(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Compute mean absolute error (MAE) loss."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def huber(y_true: ArrayLike, y_pred: ArrayLike, delta: float = 1.0) -> float:
        """Compute Huber loss (combines MSE and MAE).

        Args:
            delta: The threshold at which to switch from quadratic to linear loss.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        error = y_true - y_pred
        condition = np.abs(error) <= delta
        squared_loss = 0.5 * error**2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return float(np.mean(np.where(condition, squared_loss, linear_loss)))

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        """Fit the linear model to the training data.

        Args:
            x (ArrayLike): Feature matrix of shape (n_samples, n_features).
            y (ArrayLike): Target values of shape (n_samples,).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(x, self.weights) + self.bias
            errors = y_predicted - y

            dw = (1 / n_samples) * np.dot(x.T, errors)
            db = (1 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        """Predict target values for the given input data.

        Args:
            x (ArrayLike): Feature matrix of shape (n_samples, n_features).

        Returns:
            NDArray[np.float64]: Predicted target values.
        """
        if self.weights.size == 0:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")

        x = np.asarray(x, dtype=np.float64)
        weights = self.weights  # Pylance hint: guaranteed not None
        return np.dot(x, weights) + self.bias

    def score(
        self,
        x: ArrayLike,
        y: ArrayLike,
        metric: Union[str, Callable[[ArrayLike, ArrayLike], float]] = "r2",
        **kwargs: Any,
    ) -> float:
        """Evaluate the model's performance.

        Args:
            x (ArrayLike): Feature matrix.
            y (ArrayLike): True target values.
            metric (str or callable): Evaluation metric ('r2', 'rmse',
                            'mse', 'mae', 'huber') or a custom function.
            kwargs: Additional keyword arguments for custom metrics
                                (e.g., delta for huber).

        Returns:
            float: Evaluation score.
        """
        y_true = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(x)

        if isinstance(metric, str):
            metric = metric.lower()
            if metric == "r2":
                return self.r2(y_true, y_pred)
            elif metric == "rmse":
                return self.rmse(y_true, y_pred)
            elif metric == "mse":
                return self.mse(y_true, y_pred)
            elif metric == "mae":
                return self.mae(y_true, y_pred)
            elif metric == "huber":
                delta = float(kwargs.get("delta", 1.0))
                return self.huber(y_true, y_pred, delta=delta)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        elif callable(metric):
            return metric(y_true, y_pred)
        else:
            raise TypeError("metric must be a string or a callable function")
