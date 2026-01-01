"""Metric registry for discoverable, composable verification metrics.

The registry formalizes QVL's signature contribution: identifiability and
curvature as first-class verification metrics alongside standard accuracy.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MetricMetadata:
    """Metadata for a registered verification metric.

    Attributes:
        name: Unique metric identifier
        category: Metric category (identifiability, curvature, robustness, performance)
        description: Human-readable description
        range: Expected value range (e.g., '[0, 1]', 'R+')
        warning_threshold: Optional threshold for verification warnings
        warning_condition: Optional condition ('lt', 'gt') for warnings
        required: Whether metric must be computed
    """

    name: str
    category: str
    description: str
    range: str
    warning_threshold: Optional[float] = None
    warning_condition: Optional[str] = None  # 'lt' or 'gt'
    required: bool = False


class MetricRegistry:
    """Registry for discoverable, composable verification metrics.

    The registry pattern enables:
    - Discovery: List all available metrics and their metadata
    - Composition: Compute subsets of metrics based on category or requirements
    - Extensibility: Add custom metrics without modifying core code
    - Validation: Ensure required metrics are computed

    Example:
        >>> registry = MetricRegistry()
        >>> registry.register('ident_proxy', compute_fn, metadata)
        >>> result = registry.compute('ident_proxy', params, X, y)
        >>> warnings = registry.check_warnings(metrics)
    """

    def __init__(self):
        """Initialize empty metric registry."""
        self._metrics: Dict[str, Callable] = {}
        self._metadata: Dict[str, MetricMetadata] = {}

    def register(
        self,
        name: str,
        compute_fn: Callable,
        metadata: MetricMetadata,
    ) -> None:
        """Register a verification metric.

        Args:
            name: Unique metric identifier
            compute_fn: Callable that computes the metric
            metadata: Metric metadata

        Raises:
            ValueError: If metric name already registered
        """
        if name in self._metrics:
            raise ValueError(f"Metric '{name}' already registered")

        if metadata.name != name:
            raise ValueError(f"Metadata name '{metadata.name}' must match registered name '{name}'")

        self._metrics[name] = compute_fn
        self._metadata[name] = metadata

    def unregister(self, name: str) -> None:
        """Unregister a metric.

        Args:
            name: Metric identifier to remove

        Raises:
            KeyError: If metric not found
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not registered")

        del self._metrics[name]
        del self._metadata[name]

    def compute(self, name: str, *args, **kwargs) -> Any:
        """Compute a single metric.

        Args:
            name: Metric identifier
            *args: Positional arguments for metric computation
            **kwargs: Keyword arguments for metric computation

        Returns:
            Computed metric value

        Raises:
            KeyError: If metric not registered
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not registered")

        return self._metrics[name](*args, **kwargs)

    def compute_all(
        self,
        data: Dict[str, Any],
        categories: Optional[List[str]] = None,
        required_only: bool = False,
    ) -> Dict[str, Any]:
        """Compute multiple metrics.

        Args:
            data: Dictionary of common data needed by metrics
            categories: Optional list of categories to compute (None = all)
            required_only: Only compute required metrics

        Returns:
            Dictionary of computed metrics {name: value}
        """
        results = {}

        for name, metadata in self._metadata.items():
            # Filter by category
            if categories and metadata.category not in categories:
                continue

            # Filter by required
            if required_only and not metadata.required:
                continue

            # Compute metric
            try:
                results[name] = self._metrics[name](data)
            except Exception as e:
                # Store error but continue computing other metrics
                results[name] = None

        return results

    def get_metadata(self, name: str) -> MetricMetadata:
        """Get metadata for a metric.

        Args:
            name: Metric identifier

        Returns:
            Metric metadata

        Raises:
            KeyError: If metric not registered
        """
        if name not in self._metadata:
            raise KeyError(f"Metric '{name}' not registered")

        return self._metadata[name]

    def list_metrics(
        self,
        category: Optional[str] = None,
        required_only: bool = False,
    ) -> List[MetricMetadata]:
        """List all registered metrics.

        Args:
            category: Optional category filter
            required_only: Only list required metrics

        Returns:
            List of metric metadata
        """
        metrics = []

        for metadata in self._metadata.values():
            if category and metadata.category != category:
                continue

            if required_only and not metadata.required:
                continue

            metrics.append(metadata)

        return metrics

    def list_categories(self) -> List[str]:
        """List all metric categories.

        Returns:
            Sorted list of unique categories
        """
        return sorted(set(m.category for m in self._metadata.values()))

    def check_warnings(
        self,
        metrics: Dict[str, float],
        context: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Check for verification warnings based on metric thresholds.

        Args:
            metrics: Dictionary of computed metrics
            context: Optional context metrics for complex warnings

        Returns:
            List of warning dictionaries with keys:
                - metric: str
                - value: float
                - threshold: float
                - condition: str
                - message: str
        """
        warnings = []

        for name, value in metrics.items():
            if name not in self._metadata:
                continue

            metadata = self._metadata[name]

            if metadata.warning_threshold is None:
                continue

            if value is None:
                continue

            triggered = False

            if metadata.warning_condition == 'lt':
                triggered = value < metadata.warning_threshold
            elif metadata.warning_condition == 'gt':
                triggered = value > metadata.warning_threshold

            if triggered:
                warnings.append({
                    'metric': name,
                    'value': value,
                    'threshold': metadata.warning_threshold,
                    'condition': metadata.warning_condition,
                    'message': f"{metadata.name} = {value:.4f} {metadata.warning_condition} {metadata.warning_threshold} - {metadata.description}",
                })

        return warnings

    def validate_required(self, metrics: Dict[str, Any]) -> List[str]:
        """Validate that all required metrics are present.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            List of missing required metric names (empty if valid)
        """
        missing = []

        for name, metadata in self._metadata.items():
            if metadata.required and name not in metrics:
                missing.append(name)

        return missing


# Global metric registry instance
_global_registry = MetricRegistry()


def get_registry() -> MetricRegistry:
    """Get the global metric registry.

    Returns:
        Global MetricRegistry instance
    """
    return _global_registry
