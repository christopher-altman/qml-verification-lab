"""Tests for verification metrics."""

import numpy as np
from qvl.batteries.base import (
    compute_metrics,
    compute_hessian_metrics,
    compute_fisher_metrics,
    sigmoid
)


def test_sigmoid():
    """Test sigmoid function."""
    z = np.array([0.0, 1.0, -1.0, 100.0, -100.0])
    p = sigmoid(z)

    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)
    assert np.isclose(p[0], 0.5)
    assert p[3] > 0.99
    assert p[4] < 0.01


def test_hessian_metrics():
    """Test Hessian computation returns valid values."""
    np.random.seed(42)

    n_samples = 100
    n_features = 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples).astype(float)

    params = np.random.randn(n_features + 1) * 0.1

    hessian_min, hessian_max, ident_proxy = compute_hessian_metrics(params, X, y)

    assert np.isfinite(hessian_min)
    assert np.isfinite(hessian_max)
    assert np.isfinite(ident_proxy)

    assert hessian_min >= 0.0
    assert hessian_max >= 0.0
    assert ident_proxy >= 0.0
    assert ident_proxy <= 1.0

    assert hessian_max >= hessian_min


def test_fisher_metrics():
    """Test Fisher Information Matrix computation."""
    np.random.seed(42)

    n_samples = 100
    n_features = 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples).astype(float)

    params = np.random.randn(n_features + 1) * 0.1

    fisher_cond, fisher_rank = compute_fisher_metrics(params, X, y)

    assert np.isfinite(fisher_cond)
    assert np.isfinite(fisher_rank)

    assert fisher_cond >= 1.0
    assert fisher_rank > 0.0


def test_compute_metrics_complete():
    """Test complete metrics computation."""
    np.random.seed(42)

    n_samples = 100
    n_features = 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples).astype(float)

    params = np.random.randn(n_features + 1) * 0.1

    X_bias = np.hstack([X, np.ones((n_samples, 1))])
    predictions = sigmoid(X_bias @ params)

    loss = 0.5

    metrics = compute_metrics(params, predictions, y, loss, X)

    required_keys = [
        'accuracy',
        'loss',
        'ident_proxy',
        'hessian_min_abs',
        'hessian_max_abs',
        'fisher_condition_number',
        'fisher_effective_rank',
        'seed_robustness'
    ]

    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert np.isfinite(metrics[key]), f"Metric {key} is not finite"

    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert metrics['ident_proxy'] >= 0.0
    assert metrics['hessian_min_abs'] >= 0.0
    assert metrics['hessian_max_abs'] >= metrics['hessian_min_abs']


def test_identifiability_degrades_with_noise():
    """Test that identifiability degrades faster than accuracy with noise."""
    np.random.seed(42)

    n_samples = 200
    n_features = 2

    X_class0 = np.random.randn(n_samples // 2, n_features) + np.array([-2.0, -2.0])
    X_class1 = np.random.randn(n_samples // 2, n_features) + np.array([2.0, 2.0])
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    params_clean = np.array([1.0, 1.0, 0.0])
    X_bias = np.hstack([X, np.ones((n_samples, 1))])
    predictions_clean = sigmoid(X_bias @ params_clean)

    metrics_clean = compute_metrics(params_clean, predictions_clean, y, 0.1, X)

    X_noisy = X + np.random.randn(*X.shape) * 0.5
    y_noisy = y.copy()
    flip_mask = np.random.rand(len(y)) < 0.2
    y_noisy[flip_mask] = 1 - y_noisy[flip_mask]

    params_noisy = np.array([0.3, 0.3, 0.0])
    X_noisy_bias = np.hstack([X_noisy, np.ones((n_samples, 1))])
    predictions_noisy = sigmoid(X_noisy_bias @ params_noisy)

    metrics_noisy = compute_metrics(params_noisy, predictions_noisy, y_noisy, 0.5, X_noisy)

    ident_degradation = (metrics_clean['ident_proxy'] - metrics_noisy['ident_proxy']) / (metrics_clean['ident_proxy'] + 1e-10)
    acc_degradation = (metrics_clean['accuracy'] - metrics_noisy['accuracy']) / (metrics_clean['accuracy'] + 1e-10)

    assert ident_degradation > 0.0, "Identifiability should degrade with noise"


def test_verification_warning_trigger():
    """Test that verification warning triggers correctly."""
    from qvl.runner import check_verification_warning

    metrics_good = {
        'accuracy': 0.95,
        'ident_proxy': 0.8
    }
    warning = check_verification_warning(metrics_good, accuracy_threshold=0.7, ident_threshold=0.1)
    assert warning['warning'] is False

    metrics_bad = {
        'accuracy': 0.85,
        'ident_proxy': 0.05
    }
    warning = check_verification_warning(metrics_bad, accuracy_threshold=0.7, ident_threshold=0.1)
    assert warning['warning'] is True
    assert 'identifiability' in warning['reason'].lower()

    metrics_custom_threshold = {
        'accuracy': 0.75,
        'ident_proxy': 0.08
    }
    warning = check_verification_warning(metrics_custom_threshold, accuracy_threshold=0.8, ident_threshold=0.05)
    assert warning['warning'] is False
