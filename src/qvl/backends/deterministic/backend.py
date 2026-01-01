"""Deterministic backend - minimal working example.

This backend demonstrates the complete QVL backend interface using
only numpy. It produces perfectly reproducible results for testing
and documentation purposes.
"""

import time
import numpy as np
from typing import Dict, Any

from ...batteries import compute_metrics


class DeterministicBackend:
    """Deterministic backend for reproducible verification experiments.

    This backend uses fixed synthetic data and deterministic computations
    to demonstrate the backend interface. Perfect for:
    - Testing backend integration
    - Documentation examples
    - Reproducible benchmarks
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize deterministic backend.

        Args:
            config: Full resolved configuration dictionary
        """
        self.config = config
        self.seed = config.get('seed', 0)
        self.training_config = config.get('training', {})
        self.noise_config = config.get('noise', {})

        # Extract training parameters
        self.n_samples = self.training_config.get('n_samples', 100)
        self.n_features = self.training_config.get('n_features', 2)
        self.n_epochs = self.training_config.get('n_epochs', 100)

        # Extract noise parameters
        self.depolarizing_p = self.noise_config.get('depolarizing_p', 0.0)
        self.measurement_bitflip_p = self.noise_config.get('measurement_bitflip_p', 0.0)

        # Set seed for reproducibility
        np.random.seed(self.seed)

    def run(self) -> Dict[str, Any]:
        """Run deterministic verification experiment.

        Returns:
            Results dictionary with metrics, noise, timing
        """
        start_time = time.time()

        # Generate deterministic data
        X, y = self._generate_deterministic_data()

        # Apply noise
        X_noisy = self._apply_noise(X, self.depolarizing_p)
        y_noisy = self._apply_label_noise(y, self.measurement_bitflip_p)

        # Train model (linear separator)
        params, final_loss = self._train_linear_model(X_noisy, y_noisy)

        # Predict on clean data
        predictions = self._predict(X, params)

        # Compute metrics using QVL batteries
        metrics = compute_metrics(params, predictions, y, final_loss, X)

        wall_time = time.time() - start_time

        return {
            'metrics': metrics,
            'noise': {
                'depolarizing_p': self.depolarizing_p,
                'measurement_bitflip_p': self.measurement_bitflip_p,
                'amplitude_gamma': 0.0,
            },
            'timing': {
                'wall_time_sec': wall_time,
            },
        }

    def _generate_deterministic_data(self) -> tuple:
        """Generate deterministic synthetic data.

        Returns:
            (X, y) where X is features, y is binary labels
        """
        # Use seed-based deterministic generation
        rng = np.random.RandomState(self.seed)

        n_per_class = self.n_samples // 2

        # Class 0: centered at [-1, -1, ...]
        X_class0 = rng.randn(n_per_class, self.n_features) * 0.5 - 1.0

        # Class 1: centered at [+1, +1, ...]
        X_class1 = rng.randn(n_per_class, self.n_features) * 0.5 + 1.0

        X = np.vstack([X_class0, X_class1])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

        # Deterministic shuffle
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y

    def _apply_noise(self, X: np.ndarray, noise_level: float) -> np.ndarray:
        """Apply Gaussian noise to features.

        Args:
            X: Input features
            noise_level: Noise intensity

        Returns:
            Noisy features
        """
        if noise_level <= 0:
            return X

        rng = np.random.RandomState(self.seed + 1)
        noise = rng.randn(*X.shape) * noise_level
        return X + noise

    def _apply_label_noise(self, y: np.ndarray, flip_prob: float) -> np.ndarray:
        """Apply label flipping noise.

        Args:
            y: True labels
            flip_prob: Probability of flipping each label

        Returns:
            Noisy labels
        """
        if flip_prob <= 0:
            return y

        rng = np.random.RandomState(self.seed + 2)
        flip_mask = rng.rand(len(y)) < flip_prob
        y_noisy = y.copy()
        y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
        return y_noisy

    def _train_linear_model(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train linear model with gradient descent.

        Args:
            X: Training features
            y: Training labels

        Returns:
            (params, final_loss)
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        rng = np.random.RandomState(self.seed + 3)
        params = rng.randn(n_features + 1) * 0.01

        # Add bias column
        X_bias = np.hstack([X, np.ones((n_samples, 1))])

        learning_rate = 0.1

        # Gradient descent
        for epoch in range(self.n_epochs):
            # Forward pass
            z = X_bias @ params
            predictions = self._sigmoid(z)

            # Loss
            loss = -np.mean(
                y * np.log(predictions + 1e-10) +
                (1 - y) * np.log(1 - predictions + 1e-10)
            )

            # Gradient
            gradient = X_bias.T @ (predictions - y) / n_samples

            # Update
            params = params - learning_rate * gradient

        # Final loss
        z = X_bias @ params
        predictions = self._sigmoid(z)
        final_loss = -np.mean(
            y * np.log(predictions + 1e-10) +
            (1 - y) * np.log(1 - predictions + 1e-10)
        )

        return params, float(final_loss)

    def _predict(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features
            params: Model parameters

        Returns:
            Predicted probabilities
        """
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        z = X_bias @ params
        return self._sigmoid(z)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function.

        Args:
            z: Input values

        Returns:
            Sigmoid outputs
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
