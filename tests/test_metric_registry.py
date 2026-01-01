"""Tests for metric registry."""

import pytest
import numpy as np
from qvl.batteries import get_registry, MetricRegistry, MetricMetadata


def test_registry_registration():
    """Test metric registration and retrieval."""
    registry = MetricRegistry()

    def dummy_metric(data):
        return 42.0

    metadata = MetricMetadata(
        name='test_metric',
        category='test',
        description='A test metric',
        range='R',
        required=False,
    )

    registry.register('test_metric', dummy_metric, metadata)

    assert 'test_metric' in [m.name for m in registry.list_metrics()]
    assert registry.get_metadata('test_metric').category == 'test'


def test_registry_compute():
    """Test metric computation through registry."""
    registry = MetricRegistry()

    def add_metric(data):
        return data['a'] + data['b']

    metadata = MetricMetadata(
        name='add',
        category='test',
        description='Add two numbers',
        range='R',
    )

    registry.register('add', add_metric, metadata)

    result = registry.compute('add', {'a': 2, 'b': 3})
    assert result == 5


def test_registry_list_by_category():
    """Test filtering metrics by category."""
    registry = MetricRegistry()

    # Register metrics in different categories
    for i, cat in enumerate(['identifiability', 'curvature', 'identifiability']):
        metadata = MetricMetadata(
            name=f'metric_{i}',
            category=cat,
            description='Test',
            range='R',
        )
        registry.register(f'metric_{i}', lambda d: 0, metadata)

    ident_metrics = registry.list_metrics(category='identifiability')
    assert len(ident_metrics) == 2

    curv_metrics = registry.list_metrics(category='curvature')
    assert len(curv_metrics) == 1


def test_registry_required_validation():
    """Test validation of required metrics."""
    registry = MetricRegistry()

    metadata_required = MetricMetadata(
        name='required_metric',
        category='test',
        description='Required',
        range='R',
        required=True,
    )

    metadata_optional = MetricMetadata(
        name='optional_metric',
        category='test',
        description='Optional',
        range='R',
        required=False,
    )

    registry.register('required_metric', lambda d: 1, metadata_required)
    registry.register('optional_metric', lambda d: 2, metadata_optional)

    # Missing required metric
    missing = registry.validate_required({'optional_metric': 2})
    assert 'required_metric' in missing

    # All required present
    missing = registry.validate_required({'required_metric': 1})
    assert len(missing) == 0


def test_registry_warnings():
    """Test warning threshold checks."""
    registry = MetricRegistry()

    metadata_low = MetricMetadata(
        name='low_is_bad',
        category='test',
        description='Low values trigger warning',
        range='[0, 1]',
        warning_threshold=0.1,
        warning_condition='lt',
    )

    metadata_high = MetricMetadata(
        name='high_is_bad',
        category='test',
        description='High values trigger warning',
        range='R+',
        warning_threshold=1000.0,
        warning_condition='gt',
    )

    registry.register('low_is_bad', lambda d: 0, metadata_low)
    registry.register('high_is_bad', lambda d: 0, metadata_high)

    # Trigger low warning
    warnings = registry.check_warnings({'low_is_bad': 0.05, 'high_is_bad': 500})
    assert len(warnings) == 1
    assert warnings[0]['metric'] == 'low_is_bad'

    # Trigger high warning
    warnings = registry.check_warnings({'low_is_bad': 0.5, 'high_is_bad': 2000})
    assert len(warnings) == 1
    assert warnings[0]['metric'] == 'high_is_bad'

    # No warnings
    warnings = registry.check_warnings({'low_is_bad': 0.5, 'high_is_bad': 500})
    assert len(warnings) == 0


def test_global_registry_core_metrics():
    """Test that core metrics are registered in global registry."""
    registry = get_registry()

    # Check identifiability metrics
    ident_metrics = registry.list_metrics(category='identifiability')
    ident_names = [m.name for m in ident_metrics]
    assert 'ident_proxy' in ident_names
    assert 'fisher_condition_number' in ident_names
    assert 'fisher_effective_rank' in ident_names

    # Check curvature metrics
    curv_metrics = registry.list_metrics(category='curvature')
    curv_names = [m.name for m in curv_metrics]
    assert 'curvature_proxy' in curv_names
    assert 'hessian_trace' in curv_names
    assert 'gradient_variance' in curv_names

    # Check robustness metrics
    robust_metrics = registry.list_metrics(category='robustness')
    robust_names = [m.name for m in robust_metrics]
    assert 'noise_robustness' in robust_names
    assert 'seed_robustness' in robust_names

    # Check performance metrics
    perf_metrics = registry.list_metrics(category='performance')
    perf_names = [m.name for m in perf_metrics]
    assert 'accuracy' in perf_names
    assert 'loss' in perf_names


def test_identifiability_proxy_computation():
    """Test identifiability proxy metric computation."""
    registry = get_registry()

    # Well-conditioned case
    data = {'hessian_min': 1.0, 'hessian_max': 2.0}
    ident = registry.compute('ident_proxy', data)
    assert 0.4 < ident < 0.6  # Should be ~0.5

    # Ill-conditioned case
    data = {'hessian_min': 0.01, 'hessian_max': 100.0}
    ident = registry.compute('ident_proxy', data)
    assert ident < 0.001  # Should be very small


def test_curvature_proxy_computation():
    """Test curvature proxy metric computation."""
    registry = get_registry()

    # Use Hessian max
    data = {'hessian_max': 5.0, 'gradient_variance': 1.0}
    curv = registry.compute('curvature_proxy', data)
    assert curv == 5.0

    # Fallback to gradient variance
    data = {'gradient_variance': 2.0}
    curv = registry.compute('curvature_proxy', data)
    assert curv == 2.0


def test_fisher_condition_computation():
    """Test Fisher condition number computation."""
    registry = get_registry()

    data = {'hessian_min': 1.0, 'hessian_max': 100.0}
    cond = registry.compute('fisher_condition_number', data)
    assert cond == 100.0


def test_accuracy_computation():
    """Test accuracy metric computation."""
    registry = get_registry()

    # Perfect predictions
    data = {
        'predictions': np.array([0.9, 0.1, 0.8, 0.2]),
        'targets': np.array([1, 0, 1, 0]),
    }
    acc = registry.compute('accuracy', data)
    assert acc == 1.0

    # Half correct
    data = {
        'predictions': np.array([0.9, 0.9, 0.1, 0.1]),
        'targets': np.array([1, 0, 1, 0]),
    }
    acc = registry.compute('accuracy', data)
    assert acc == 0.5


def test_loss_computation():
    """Test loss metric computation."""
    registry = get_registry()

    # Direct loss
    data = {'loss': 0.5}
    loss = registry.compute('loss', data)
    assert loss == 0.5

    # Compute from predictions
    data = {
        'predictions': np.array([0.9, 0.1]),
        'targets': np.array([1, 0]),
    }
    loss = registry.compute('loss', data)
    assert loss > 0  # Should be positive binary cross-entropy


def test_list_categories():
    """Test listing all metric categories."""
    registry = get_registry()

    categories = registry.list_categories()
    assert 'identifiability' in categories
    assert 'curvature' in categories
    assert 'robustness' in categories
    assert 'performance' in categories


def test_required_metrics():
    """Test that required metrics are flagged correctly."""
    registry = get_registry()

    required = registry.list_metrics(required_only=True)
    required_names = [m.name for m in required]

    assert 'ident_proxy' in required_names
    assert 'accuracy' in required_names
    assert 'loss' in required_names


def test_metric_metadata_warnings():
    """Test that warning thresholds are set correctly."""
    registry = get_registry()

    # ident_proxy should warn when < 0.1
    ident_meta = registry.get_metadata('ident_proxy')
    assert ident_meta.warning_threshold == 0.1
    assert ident_meta.warning_condition == 'lt'

    # fisher_condition_number should warn when > 1000
    fisher_meta = registry.get_metadata('fisher_condition_number')
    assert fisher_meta.warning_threshold == 1000.0
    assert fisher_meta.warning_condition == 'gt'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
