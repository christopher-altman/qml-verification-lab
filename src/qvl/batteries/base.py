"""Base metrics computation."""

import numpy as np
from typing import Dict, Any


def compute_metrics(
    model_params: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    loss: float,
    X: np.ndarray
) -> Dict[str, float]:
    """Compute verification metrics.

    Args:
        model_params: Model parameters (weights)
        predictions: Model predictions
        targets: Ground truth targets
        loss: Training loss
        X: Input features

    Returns:
        Dictionary of metrics
    """
    accuracy = float(np.mean((predictions > 0.5) == targets))

    hessian_min, hessian_max, ident_proxy = compute_hessian_metrics(model_params, X, targets)

    fisher_cond, fisher_rank = compute_fisher_metrics(model_params, X, targets)

    return {
        'accuracy': accuracy,
        'loss': loss,
        'ident_proxy': ident_proxy,
        'hessian_min_abs': hessian_min,
        'hessian_max_abs': hessian_max,
        'fisher_condition_number': fisher_cond,
        'fisher_effective_rank': fisher_rank,
        'seed_robustness': 0.0,
    }


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def compute_hessian_metrics(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple:
    """Compute Hessian-based metrics for logistic regression.

    For logistic regression, the Hessian of the negative log-likelihood is:
    H = (1/n) * X^T @ diag(p(1-p)) @ X
    where p = sigmoid(X @ params)

    Returns:
        (hessian_min_abs, hessian_max_abs, ident_proxy)
    """
    eps = 1e-10
    n_samples = X.shape[0]

    if n_samples < 2:
        return 0.0, 0.0, 0.0

    X_bias = np.hstack([X, np.ones((n_samples, 1))])

    z = X_bias @ params
    p = sigmoid(z)

    p_clipped = np.clip(p * (1 - p), eps, 1.0)
    W = np.diag(p_clipped)

    H = (X_bias.T @ W @ X_bias) / n_samples

    try:
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.abs(eigvals)
        eigvals = eigvals[eigvals > eps]

        if len(eigvals) == 0:
            return 0.0, 0.0, 0.0

        hessian_min = float(eigvals[0])
        hessian_max = float(eigvals[-1])

        cond_number = hessian_max / (hessian_min + eps)
        ident_proxy = float(hessian_min / (hessian_max + eps))

        return hessian_min, hessian_max, ident_proxy

    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0


def compute_fisher_metrics(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple:
    """Compute Fisher Information Matrix metrics for logistic regression.

    For logistic regression, the Fisher Information Matrix equals the Hessian:
    F = (1/n) * X^T @ diag(p(1-p)) @ X

    Returns:
        (fisher_condition_number, fisher_effective_rank)
    """
    eps = 1e-10
    n_samples = X.shape[0]

    if n_samples < 2:
        return 1.0, 1.0

    X_bias = np.hstack([X, np.ones((n_samples, 1))])

    z = X_bias @ params
    p = sigmoid(z)

    p_clipped = np.clip(p * (1 - p), eps, 1.0)
    W = np.diag(p_clipped)

    F = (X_bias.T @ W @ X_bias) / n_samples

    try:
        eigvals = np.linalg.eigvalsh(F)
        eigvals = eigvals[eigvals > eps]

        if len(eigvals) == 0:
            return 1.0, 0.0

        cond_number = float(eigvals[-1] / (eigvals[0] + eps))

        eigvals_normalized = eigvals / np.sum(eigvals)
        entropy = -np.sum(eigvals_normalized * np.log(eigvals_normalized + eps))
        effective_rank = float(np.exp(entropy))

        return cond_number, effective_rank

    except np.linalg.LinAlgError:
        return 1.0, 1.0


def compute_seed_robustness(results: list) -> float:
    """Compute seed robustness from multiple runs at same noise point.

    Args:
        results: List of result dictionaries with same noise settings

    Returns:
        Standard deviation of accuracy across seeds
    """
    if len(results) < 2:
        return 0.0

    accuracies = [r['metrics']['accuracy'] for r in results]
    return float(np.std(accuracies))
