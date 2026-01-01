"""Toy backend implementation."""

import time
import numpy as np
from typing import Dict, Any

from ...batteries import compute_metrics


class ToyBackend:
    """Toy backend using simple logistic regression on synthetic data."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize toy backend.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.seed = config.get('seed', 0)
        self.training_config = config.get('training', {})
        self.noise_config = config.get('noise', {})

        self.n_samples = self.training_config.get('n_samples', 100)
        self.n_features = self.training_config.get('n_features', 2)
        self.n_epochs = self.training_config.get('n_epochs', 100)
        self.learning_rate = self.training_config.get('learning_rate', 0.1)

        self.depolarizing_p = self.noise_config.get('depolarizing_p', 0.0)
        self.measurement_bitflip_p = self.noise_config.get('measurement_bitflip_p', 0.0)
        self.amplitude_gamma = self.noise_config.get('amplitude_gamma', 0.0)

        np.random.seed(self.seed)

    def run(self) -> Dict[str, Any]:
        """Run toy experiment.

        Returns:
            Results dictionary
        """
        start_time = time.time()

        X, y = self._generate_data()

        X_noisy = self._apply_feature_noise(X)
        y_noisy = self._apply_label_noise(y)

        params, final_loss = self._train(X_noisy, y_noisy)

        predictions = self._predict(X, params)

        metrics = compute_metrics(params, predictions, y, final_loss, X)

        wall_time = time.time() - start_time

        return {
            'metrics': metrics,
            'timing': {
                'wall_time_sec': wall_time,
            },
            'noise': {
                'depolarizing_p': self.depolarizing_p,
                'measurement_bitflip_p': self.measurement_bitflip_p,
                'amplitude_gamma': self.amplitude_gamma,
            },
        }

    def _generate_data(self) -> tuple:
        """Generate synthetic binary classification dataset."""
        n_per_class = self.n_samples // 2

        X_class0 = np.random.randn(n_per_class, self.n_features) + np.array([-1.0] * self.n_features)
        X_class1 = np.random.randn(n_per_class, self.n_features) + np.array([1.0] * self.n_features)

        X = np.vstack([X_class0, X_class1])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y

    def _apply_feature_noise(self, X: np.ndarray) -> np.ndarray:
        """Apply depolarizing noise to features."""
        if self.depolarizing_p <= 0:
            return X

        noise = np.random.randn(*X.shape) * self.depolarizing_p
        return X + noise

    def _apply_label_noise(self, y: np.ndarray) -> np.ndarray:
        """Apply measurement bitflip noise to labels."""
        if self.measurement_bitflip_p <= 0:
            return y

        flip_mask = np.random.rand(len(y)) < self.measurement_bitflip_p
        y_noisy = y.copy()
        y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
        return y_noisy

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _train(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train logistic regression model.

        Returns:
            (params, final_loss)
        """
        n_samples, n_features = X.shape

        params = np.random.randn(n_features + 1) * 0.01

        X_bias = np.hstack([X, np.ones((n_samples, 1))])

        for epoch in range(self.n_epochs):
            z = X_bias @ params
            predictions = self._sigmoid(z)

            loss = -np.mean(y * np.log(predictions + 1e-10) + (1 - y) * np.log(1 - predictions + 1e-10))

            gradient = X_bias.T @ (predictions - y) / n_samples

            if self.amplitude_gamma > 0:
                gradient += self.amplitude_gamma * params

            params = params - self.learning_rate * gradient

        z = X_bias @ params
        predictions = self._sigmoid(z)
        final_loss = -np.mean(y * np.log(predictions + 1e-10) + (1 - y) * np.log(1 - predictions + 1e-10))

        return params, float(final_loss)

    def _predict(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features
            params: Model parameters

        Returns:
            Predictions (probabilities)
        """
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        z = X_bias @ params
        return self._sigmoid(z)
