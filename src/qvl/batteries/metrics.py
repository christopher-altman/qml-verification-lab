"""Core verification metrics with registry integration.

This module implements QVL's signature contribution: identifiability and
curvature as first-class verification metrics.
"""

import numpy as np
from typing import Dict, Any

from .registry import get_registry, MetricMetadata


# ============================================================================
# Identifiability Metrics
# ============================================================================

def compute_identifiability_proxy(data: Dict[str, Any]) -> float:
    """Compute identifiability proxy from Hessian eigenvalues.

    Identifiability measures whether model parameters can be uniquely
    determined from observations. Low identifiability indicates many
    parameter settings produce similar outputs (non-identifiable model).

    Args:
        data: Dictionary containing 'hessian_min' and 'hessian_max'

    Returns:
        Identifiability proxy in [0, 1], where:
            1.0 = perfectly identifiable (Hessian well-conditioned)
            0.0 = non-identifiable (Hessian ill-conditioned)
    """
    hessian_min = data.get('hessian_min', 0.0)
    hessian_max = data.get('hessian_max', 1.0)

    if hessian_max < 1e-10:
        return 0.0

    return float(hessian_min / (hessian_max + 1e-10))


def compute_fisher_condition(data: Dict[str, Any]) -> float:
    """Compute Fisher Information Matrix condition number.

    High condition number indicates ill-conditioned parameter space,
    making parameter estimation unreliable.

    Args:
        data: Dictionary containing 'hessian_min' and 'hessian_max'
              (Fisher = Hessian for maximum likelihood)

    Returns:
        Condition number (ratio of max/min eigenvalues)
    """
    hessian_min = data.get('hessian_min', 1.0)
    hessian_max = data.get('hessian_max', 1.0)

    if hessian_min < 1e-10:
        return 1e6  # Cap at large value for numerical stability

    return float(hessian_max / hessian_min)


def compute_fisher_rank(data: Dict[str, Any]) -> float:
    """Compute effective rank of Fisher Information Matrix.

    Effective rank reveals how many parameters are actually determined
    by the data (may be less than theoretical parameter count).

    Args:
        data: Dictionary containing 'hessian_eigenvalues' (optional)
              Falls back to theoretical rank from 'n_params'

    Returns:
        Effective rank (entropy-based)
    """
    eigvals = data.get('hessian_eigenvalues', None)

    if eigvals is None or len(eigvals) == 0:
        # Fallback: use theoretical rank
        return float(data.get('n_params', 1.0))

    eigvals = np.array(eigvals)
    eigvals = eigvals[eigvals > 1e-10]

    if len(eigvals) == 0:
        return 0.0

    # Normalize eigenvalues
    eigvals_norm = eigvals / np.sum(eigvals)

    # Compute entropy-based effective rank
    entropy = -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-10))
    effective_rank = float(np.exp(entropy))

    return effective_rank


# ============================================================================
# Curvature Metrics
# ============================================================================

def compute_curvature_proxy(data: Dict[str, Any]) -> float:
    """Compute loss landscape curvature proxy.

    High curvature indicates sharp minima (poor generalization),
    low curvature indicates flat minima (better generalization).

    Args:
        data: Dictionary containing 'hessian_max' or 'gradient_variance'

    Returns:
        Curvature proxy (positive real number)
    """
    # Primary: use Hessian trace proxy (max eigenvalue)
    hessian_max = data.get('hessian_max', None)

    if hessian_max is not None and hessian_max > 0:
        return float(hessian_max)

    # Fallback: use gradient variance
    grad_var = data.get('gradient_variance', 0.0)
    return float(grad_var)


def compute_hessian_trace(data: Dict[str, Any]) -> float:
    """Compute trace of Hessian matrix.

    Trace = sum of eigenvalues, measures average curvature.

    Args:
        data: Dictionary containing 'hessian_eigenvalues' or 'hessian_min'/'hessian_max'

    Returns:
        Hessian trace (sum of eigenvalues)
    """
    eigvals = data.get('hessian_eigenvalues', None)

    if eigvals is not None:
        return float(np.sum(eigvals))

    # Fallback: approximate from min/max (assumes uniform distribution)
    hessian_min = data.get('hessian_min', 0.0)
    hessian_max = data.get('hessian_max', 0.0)
    n_params = data.get('n_params', 1)

    # Linear interpolation approximation
    return float((hessian_min + hessian_max) / 2.0 * n_params)


def compute_gradient_variance(data: Dict[str, Any]) -> float:
    """Compute variance of gradients across data points.

    High variance indicates unstable gradients (potential barren plateaus).

    Args:
        data: Dictionary containing 'gradients' (array of per-sample gradients)

    Returns:
        Gradient variance
    """
    gradients = data.get('gradients', None)

    if gradients is None:
        return 0.0

    gradients = np.array(gradients)

    if gradients.ndim == 1:
        return float(np.var(gradients))

    # For multi-dimensional gradients, compute mean variance across dimensions
    return float(np.mean(np.var(gradients, axis=0)))


# ============================================================================
# Robustness Metrics
# ============================================================================

def compute_noise_robustness(data: Dict[str, Any]) -> float:
    """Compute robustness to noise perturbations.

    Measures how much accuracy degrades under noise.

    Args:
        data: Dictionary containing 'accuracy_clean' and 'accuracy_noisy'

    Returns:
        Robustness score in [0, 1] where 1.0 = no degradation
    """
    acc_clean = data.get('accuracy_clean', 1.0)
    acc_noisy = data.get('accuracy_noisy', 0.0)

    if acc_clean < 1e-10:
        return 0.0

    return float(acc_noisy / acc_clean)


def compute_seed_robustness(data: Dict[str, Any]) -> float:
    """Compute robustness across random seeds.

    Low variance indicates stable solution, high variance indicates
    non-robustness.

    Args:
        data: Dictionary containing 'accuracy_seeds' (list of accuracies)

    Returns:
        Standard deviation of accuracy across seeds
    """
    acc_seeds = data.get('accuracy_seeds', None)

    if acc_seeds is None or len(acc_seeds) < 2:
        return 0.0

    return float(np.std(acc_seeds))


# ============================================================================
# Performance Metrics
# ============================================================================

def compute_accuracy(data: Dict[str, Any]) -> float:
    """Compute classification accuracy.

    Args:
        data: Dictionary containing 'predictions' and 'targets'

    Returns:
        Accuracy in [0, 1]
    """
    predictions = data.get('predictions', None)
    targets = data.get('targets', None)

    if predictions is None or targets is None:
        return 0.0

    predictions = np.array(predictions)
    targets = np.array(targets)

    return float(np.mean((predictions > 0.5) == targets))


def compute_loss(data: Dict[str, Any]) -> float:
    """Compute training loss.

    Args:
        data: Dictionary containing 'loss' or 'predictions'/'targets'

    Returns:
        Loss value
    """
    # Direct loss if provided
    loss = data.get('loss', None)
    if loss is not None:
        return float(loss)

    # Compute binary cross-entropy from predictions
    predictions = data.get('predictions', None)
    targets = data.get('targets', None)

    if predictions is None or targets is None:
        return 0.0

    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
    targets = np.array(targets)

    bce = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return float(bce)


# ============================================================================
# Registry Initialization
# ============================================================================

def register_core_metrics() -> None:
    """Register core verification metrics with the global registry."""
    registry = get_registry()

    # Identifiability metrics
    registry.register(
        'ident_proxy',
        compute_identifiability_proxy,
        MetricMetadata(
            name='ident_proxy',
            category='identifiability',
            description='Identifiability proxy (min/max Hessian eigenvalue ratio)',
            range='[0, 1]',
            warning_threshold=0.1,
            warning_condition='lt',
            required=True,
        ),
    )

    registry.register(
        'fisher_condition_number',
        compute_fisher_condition,
        MetricMetadata(
            name='fisher_condition_number',
            category='identifiability',
            description='Fisher Information Matrix condition number',
            range='R+ (1 = best)',
            warning_threshold=1000.0,
            warning_condition='gt',
            required=False,
        ),
    )

    registry.register(
        'fisher_effective_rank',
        compute_fisher_rank,
        MetricMetadata(
            name='fisher_effective_rank',
            category='identifiability',
            description='Effective rank of Fisher Information Matrix',
            range='R+',
            warning_threshold=None,
            warning_condition=None,
            required=False,
        ),
    )

    # Curvature metrics
    registry.register(
        'curvature_proxy',
        compute_curvature_proxy,
        MetricMetadata(
            name='curvature_proxy',
            category='curvature',
            description='Loss landscape curvature (Hessian max eigenvalue)',
            range='R+',
            warning_threshold=None,
            warning_condition=None,
            required=False,
        ),
    )

    registry.register(
        'hessian_trace',
        compute_hessian_trace,
        MetricMetadata(
            name='hessian_trace',
            category='curvature',
            description='Trace of Hessian matrix (average curvature)',
            range='R',
            warning_threshold=None,
            warning_condition=None,
            required=False,
        ),
    )

    registry.register(
        'gradient_variance',
        compute_gradient_variance,
        MetricMetadata(
            name='gradient_variance',
            category='curvature',
            description='Variance of gradients across data points',
            range='R+',
            warning_threshold=None,
            warning_condition=None,
            required=False,
        ),
    )

    # Robustness metrics
    registry.register(
        'noise_robustness',
        compute_noise_robustness,
        MetricMetadata(
            name='noise_robustness',
            category='robustness',
            description='Robustness to noise perturbations (accuracy ratio)',
            range='[0, 1]',
            warning_threshold=None,
            warning_condition=None,
            required=False,
        ),
    )

    registry.register(
        'seed_robustness',
        compute_seed_robustness,
        MetricMetadata(
            name='seed_robustness',
            category='robustness',
            description='Robustness across random seeds (accuracy std)',
            range='R+',
            warning_threshold=0.1,
            warning_condition='gt',
            required=False,
        ),
    )

    # Performance metrics
    registry.register(
        'accuracy',
        compute_accuracy,
        MetricMetadata(
            name='accuracy',
            category='performance',
            description='Classification accuracy',
            range='[0, 1]',
            warning_threshold=None,
            warning_condition=None,
            required=True,
        ),
    )

    registry.register(
        'loss',
        compute_loss,
        MetricMetadata(
            name='loss',
            category='performance',
            description='Training loss',
            range='R+',
            warning_threshold=None,
            warning_condition=None,
            required=True,
        ),
    )


# Auto-register core metrics on import
register_core_metrics()
