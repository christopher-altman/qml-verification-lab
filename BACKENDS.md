# How to Add a Backend

Add custom backends to QVL in **under 60 seconds**. Backends implement quantum or classical models for verification experiments.

## Backend Interface

Every backend implements two methods:

```python
class MyBackend:
    def __init__(self, config: Dict[str, Any]):
        """Initialize backend with experiment configuration.

        Args:
            config: Full resolved configuration dictionary
        """
        pass

    def run(self) -> Dict[str, Any]:
        """Run experiment and return results.

        Returns:
            Dictionary with required keys:
                - metrics: dict of verification metrics
                - noise: dict of applied noise parameters
                - timing: dict with wall_time_sec
        """
        pass
```

## Lifecycle

1. **Initialization**: `__init__(config)` receives the full resolved config
2. **Execution**: `run()` performs training/inference
3. **Return**: Return standardized metrics dictionary

## Configuration Access

Backends receive a config dictionary with standard keys:

```python
config = {
    'experiment_id': str,           # Unique experiment identifier
    'backend': str,                 # Backend name (e.g., 'toy', 'pennylane')
    'task': str,                    # 'classification' or 'regression'
    'seed': int,                    # Random seed for this run
    'training': {
        'n_samples': int,           # Dataset size
        'n_features': int,          # Input dimensionality
        'n_epochs': int,            # Training iterations
        'learning_rate': float,     # Optimizer step size
        'train_frac': float,        # Train/test split ratio
        'n_layers': int,            # Model depth (circuit or classical)
    },
    'noise': {
        'depolarizing_p': float,    # Feature/gate noise [0, 1]
        'measurement_bitflip_p': float,  # Label/measurement noise [0, 1]
        'amplitude_gamma': float,   # Regularization/damping [0, 1]
    },
    'verification': {
        'accuracy_threshold': float,  # Warning threshold for accuracy
        'ident_threshold': float,     # Warning threshold for identifiability
    },
}
```

## Expected Output

`run()` must return:

```python
{
    'metrics': {
        'accuracy': float,                      # Required: Test accuracy [0, 1]
        'loss': float,                          # Required: Final training loss
        'ident_proxy': float,                   # Required: Identifiability proxy [0, 1]
        'hessian_min_abs': float,               # Optional: Min Hessian eigenvalue
        'hessian_max_abs': float,               # Optional: Max Hessian eigenvalue
        'fisher_condition_number': float,       # Optional: Fisher matrix condition
        'fisher_effective_rank': float,         # Optional: Effective rank
        'seed_robustness': float,               # Optional: Cross-seed variance
    },
    'noise': {
        'depolarizing_p': float,                # Actual noise applied
        'measurement_bitflip_p': float,
        'amplitude_gamma': float,
    },
    'timing': {
        'wall_time_sec': float,                 # Total execution time
    },
}
```

**Required metrics**: `accuracy`, `loss`, `ident_proxy`
**Optional metrics**: All others (use `0.0` if not computed)

## Minimal Template Backend

Create `src/qvl/backends/mybackend/backend.py`:

```python
"""My custom backend implementation."""

import time
import numpy as np
from typing import Dict, Any


class MyBackend:
    """Template backend for QVL verification experiments."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize backend.

        Args:
            config: Full resolved configuration
        """
        self.config = config
        self.seed = config.get('seed', 0)
        self.training_config = config.get('training', {})
        self.noise_config = config.get('noise', {})

        # Extract training parameters
        self.n_samples = self.training_config.get('n_samples', 100)
        self.n_epochs = self.training_config.get('n_epochs', 100)
        self.learning_rate = self.training_config.get('learning_rate', 0.1)

        # Extract noise parameters
        self.depolarizing_p = self.noise_config.get('depolarizing_p', 0.0)
        self.measurement_bitflip_p = self.noise_config.get('measurement_bitflip_p', 0.0)
        self.amplitude_gamma = self.noise_config.get('amplitude_gamma', 0.0)

        np.random.seed(self.seed)

    def run(self) -> Dict[str, Any]:
        """Run experiment.

        Returns:
            Results dictionary with metrics, noise, timing
        """
        start_time = time.time()

        # TODO: Implement your model training here
        # 1. Generate or load data
        # 2. Apply noise (depolarizing_p, measurement_bitflip_p, amplitude_gamma)
        # 3. Train model
        # 4. Compute predictions
        # 5. Calculate verification metrics

        # Placeholder metrics (replace with real computation)
        accuracy = 0.5  # Replace with actual accuracy
        loss = 1.0      # Replace with actual loss
        ident_proxy = 0.0  # Replace with identifiability computation

        wall_time = time.time() - start_time

        return {
            'metrics': {
                'accuracy': accuracy,
                'loss': loss,
                'ident_proxy': ident_proxy,
                'hessian_min_abs': 0.0,
                'hessian_max_abs': 0.0,
                'fisher_condition_number': 1.0,
                'fisher_effective_rank': 1.0,
                'seed_robustness': 0.0,
            },
            'noise': {
                'depolarizing_p': self.depolarizing_p,
                'measurement_bitflip_p': self.measurement_bitflip_p,
                'amplitude_gamma': self.amplitude_gamma,
            },
            'timing': {
                'wall_time_sec': wall_time,
            },
        }
```

Create `src/qvl/backends/mybackend/__init__.py`:

```python
"""My backend module."""

# Optional: Add dependency check
try:
    import my_required_library
    HAS_MY_LIBRARY = True
except ImportError:
    HAS_MY_LIBRARY = False

if not HAS_MY_LIBRARY:
    raise ImportError(
        "my_required_library is required for this backend. "
        "Install with: pip install my_required_library"
    )
```

## Register Backend

Edit `src/qvl/backends/__init__.py`:

```python
from .toy.backend import ToyBackend

BACKENDS = {
    'toy': ToyBackend,
}

# Add your backend
try:
    from .mybackend import HAS_MY_LIBRARY
    if HAS_MY_LIBRARY:
        from .mybackend.backend import MyBackend
        BACKENDS['mybackend'] = MyBackend
except ImportError:
    pass
```

## Add Example Config

Create `examples/mybackend_smoke.yaml`:

```yaml
experiment_id: mybackend_test
backend: mybackend
task: classification

training:
  n_samples: 100
  n_features: 2
  n_epochs: 50
  learning_rate: 0.1

noise:
  depolarizing_p: 0.1
  measurement_bitflip_p: 0.05
  amplitude_gamma: 0.01
```

## Test Your Backend

```bash
# Single run
python -m qvl run --config examples/mybackend_smoke.yaml --output-dir artifacts/

# Parameter sweep
python -m qvl sweep --config examples/mybackend_sweep.yaml --output-dir artifacts/ --seeds 0,1,2
```

## Verification Metrics Guide

### Required Metrics

**`accuracy`**: Standard test accuracy
- Binary classification: fraction of correct predictions
- Regression: 1 - normalized MSE

**`loss`**: Final training loss
- Binary cross-entropy for classification
- MSE for regression

**`ident_proxy`**: Identifiability proxy in [0, 1]
- Ratio of min/max Hessian eigenvalues
- Higher = more identifiable
- < 0.1 triggers verification warning

### Optional Metrics

**`hessian_min_abs`, `hessian_max_abs`**: Loss landscape geometry
- Use finite differences if analytic Hessian unavailable
- Reveals parameter space conditioning

**`fisher_condition_number`**: Parameter determinability
- Ratio of max/min Fisher eigenvalues
- > 1000 indicates ill-conditioning

**`fisher_effective_rank`**: Active parameter dimensions
- Entropy-based effective rank
- Compare to theoretical parameter count

**`seed_robustness`**: Stability across seeds
- Standard deviation of accuracy across multiple seeds
- Requires aggregation across runs

## Noise Implementation Patterns

### Depolarizing Noise (`depolarizing_p`)
Feature/gate noise that randomizes inputs or quantum states:

```python
# Classical feature noise
X_noisy = X + np.random.randn(*X.shape) * self.depolarizing_p

# Quantum gate noise (PennyLane example)
qml.DepolarizingChannel(self.depolarizing_p, wires=qubit)
```

### Measurement Bitflip Noise (`measurement_bitflip_p`)
Label/measurement noise that flips outputs:

```python
# Classical label noise
flip_mask = np.random.rand(len(y)) < self.measurement_bitflip_p
y_noisy = y.copy()
y_noisy[flip_mask] = 1 - y_noisy[flip_mask]

# Quantum measurement noise
if np.random.rand() < self.measurement_bitflip_p:
    prediction = 1 - prediction
```

### Amplitude Damping (`amplitude_gamma`)
Regularization or quantum amplitude damping:

```python
# Classical regularization
gradient += self.amplitude_gamma * params

# Quantum amplitude damping (PennyLane)
qml.AmplitudeDamping(self.amplitude_gamma, wires=qubit)
```

## Real Backend Examples

**Toy backend** (`src/qvl/backends/toy/backend.py`): Logistic regression with analytic Fisher/Hessian
**PennyLane backend** (`src/qvl/backends/pennylane/backend.py`): 2-qubit VQC with quantum noise channels

Study these for full reference implementations.

## Summary

1. Create backend class with `__init__(config)` and `run() -> dict`
2. Return required metrics: `accuracy`, `loss`, `ident_proxy`
3. Register in `backends/__init__.py`
4. Add example config
5. Test with `python -m qvl run`

**Total time: < 60 seconds** for minimal stub, 5-10 minutes for working backend.
