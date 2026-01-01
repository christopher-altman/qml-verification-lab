#!/usr/bin/env python3
"""Demonstration of the metric registry for composable verification metrics.

This example shows how to:
1. Discover available metrics
2. Compute metrics using the registry
3. Check for verification warnings
4. Register custom metrics
"""

import numpy as np
from qvl.batteries import get_registry, MetricMetadata


def main():
    """Run metric registry demonstration."""
    print("=" * 70)
    print("QVL Metric Registry Demo")
    print("=" * 70)
    print()

    registry = get_registry()

    # ========================================================================
    # 1. Discovery: List all available metrics
    # ========================================================================

    print("1. Available Metric Categories:")
    print("-" * 70)
    categories = registry.list_categories()
    for cat in categories:
        metrics = registry.list_metrics(category=cat)
        print(f"\n{cat.upper()} ({len(metrics)} metrics):")
        for m in metrics:
            required = " [REQUIRED]" if m.required else ""
            print(f"  - {m.name}: {m.description}{required}")
            print(f"    Range: {m.range}")
            if m.warning_threshold:
                print(f"    Warning: {m.warning_condition} {m.warning_threshold}")

    # ========================================================================
    # 2. Computation: Calculate metrics
    # ========================================================================

    print("\n\n2. Computing Metrics:")
    print("-" * 70)

    # Prepare mock data
    data_well_conditioned = {
        'hessian_min': 1.0,
        'hessian_max': 2.0,
        'predictions': np.array([0.9, 0.1, 0.8, 0.2]),
        'targets': np.array([1, 0, 1, 0]),
        'loss': 0.1,
    }

    data_ill_conditioned = {
        'hessian_min': 0.01,
        'hessian_max': 100.0,
        'predictions': np.array([0.9, 0.1, 0.8, 0.2]),
        'targets': np.array([1, 0, 1, 0]),
        'loss': 0.1,
    }

    # Compute identifiability metrics
    print("\nWell-conditioned case (hessian_max/hessian_min = 2):")
    ident = registry.compute('ident_proxy', data_well_conditioned)
    fisher_cond = registry.compute('fisher_condition_number', data_well_conditioned)
    print(f"  Identifiability proxy: {ident:.4f}")
    print(f"  Fisher condition number: {fisher_cond:.2f}")

    print("\nIll-conditioned case (hessian_max/hessian_min = 10000):")
    ident = registry.compute('ident_proxy', data_ill_conditioned)
    fisher_cond = registry.compute('fisher_condition_number', data_ill_conditioned)
    print(f"  Identifiability proxy: {ident:.4f}")
    print(f"  Fisher condition number: {fisher_cond:.2f}")

    # Compute curvature metrics
    print("\nCurvature metrics:")
    curv = registry.compute('curvature_proxy', data_well_conditioned)
    print(f"  Curvature proxy: {curv:.4f}")

    # Compute performance metrics
    print("\nPerformance metrics:")
    acc = registry.compute('accuracy', data_well_conditioned)
    loss = registry.compute('loss', data_well_conditioned)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Loss: {loss:.4f}")

    # ========================================================================
    # 3. Warnings: Check verification thresholds
    # ========================================================================

    print("\n\n3. Verification Warnings:")
    print("-" * 70)

    # Good case: no warnings
    good_metrics = {
        'ident_proxy': 0.5,
        'fisher_condition_number': 100.0,
        'accuracy': 0.95,
    }

    warnings = registry.check_warnings(good_metrics)
    print(f"\nGood metrics (ident={good_metrics['ident_proxy']}, fisher_cond={good_metrics['fisher_condition_number']}):")
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w['message']}")
    else:
        print("  No warnings - model is well-verified!")

    # Bad case: warnings triggered
    bad_metrics = {
        'ident_proxy': 0.05,  # Below 0.1 threshold
        'fisher_condition_number': 5000.0,  # Above 1000 threshold
        'accuracy': 0.85,
    }

    warnings = registry.check_warnings(bad_metrics)
    print(f"\nBad metrics (ident={bad_metrics['ident_proxy']}, fisher_cond={bad_metrics['fisher_condition_number']}):")
    if warnings:
        for w in warnings:
            print(f"  ⚠️  WARNING: {w['message']}")
    else:
        print("  No warnings")

    # ========================================================================
    # 4. Validation: Check required metrics
    # ========================================================================

    print("\n\n4. Required Metric Validation:")
    print("-" * 70)

    # Missing required metrics
    incomplete_metrics = {
        'accuracy': 0.9,
        # Missing: ident_proxy, loss
    }

    missing = registry.validate_required(incomplete_metrics)
    if missing:
        print(f"\n❌ Missing required metrics: {', '.join(missing)}")
    else:
        print("\n✅ All required metrics present")

    # Complete metrics
    complete_metrics = {
        'accuracy': 0.9,
        'loss': 0.2,
        'ident_proxy': 0.5,
    }

    missing = registry.validate_required(complete_metrics)
    if missing:
        print(f"\n❌ Missing required metrics: {', '.join(missing)}")
    else:
        print("\n✅ All required metrics present")

    # ========================================================================
    # 5. Extensibility: Register custom metric
    # ========================================================================

    print("\n\n5. Custom Metric Registration:")
    print("-" * 70)

    def compute_my_metric(data):
        """Custom verification metric."""
        acc = data.get('accuracy', 0.0)
        ident = data.get('ident_proxy', 0.0)
        # Custom heuristic: product of accuracy and identifiability
        return acc * ident

    metadata = MetricMetadata(
        name='acc_ident_product',
        category='custom',
        description='Product of accuracy and identifiability',
        range='[0, 1]',
        warning_threshold=0.5,
        warning_condition='lt',
        required=False,
    )

    registry.register('acc_ident_product', compute_my_metric, metadata)

    print("\nRegistered custom metric: acc_ident_product")
    print("Computing on sample data:")

    custom_data = {
        'accuracy': 0.9,
        'ident_proxy': 0.7,
    }

    custom_value = registry.compute('acc_ident_product', custom_data)
    print(f"  acc_ident_product = {custom_value:.4f}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total metrics registered: {len(registry.list_metrics())}")
    print(f"Categories: {', '.join(registry.list_categories())}")
    print(f"Required metrics: {len(registry.list_metrics(required_only=True))}")
    print("\nThe metric registry makes identifiability and curvature")
    print("first-class citizens in QML verification.")
    print("=" * 70)


if __name__ == '__main__':
    main()
