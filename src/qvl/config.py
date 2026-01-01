"""Configuration loading and validation."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if yaml is None:
        raise ImportError("pyyaml is required. Install with: pip install pyyaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure."""
    required_fields = ['experiment_id', 'backend', 'task']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    # Import backends dynamically to get available backends
    from .backends import BACKENDS
    valid_backends = list(BACKENDS.keys())

    if config['backend'] not in valid_backends:
        raise ValueError(
            f"Unknown backend: {config['backend']}. "
            f"Valid backends: {', '.join(valid_backends)}"
        )

    if config['task'] not in ['classification', 'regression']:
        raise ValueError(f"Unknown task: {config['task']}")


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute short hash of configuration."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def resolve_config(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Resolve configuration with seed and defaults."""
    resolved = config.copy()
    resolved['seed'] = seed

    # Add defaults if not present
    if 'noise' not in resolved:
        resolved['noise'] = {}

    noise_defaults = {
        'depolarizing_p': 0.0,
        'measurement_bitflip_p': 0.0,
        'amplitude_gamma': 0.0,
    }
    for key, default_val in noise_defaults.items():
        if key not in resolved['noise']:
            resolved['noise'][key] = default_val

    if 'training' not in resolved:
        resolved['training'] = {}

    training_defaults = {
        'n_samples': 100,
        'n_features': 2,
        'n_epochs': 100,
        'learning_rate': 0.1,
        'train_frac': 0.8,
        'batch_size': None,
        'n_layers': 2,
    }
    for key, default_val in training_defaults.items():
        if key not in resolved['training']:
            resolved['training'][key] = default_val

    if 'verification' not in resolved:
        resolved['verification'] = {}

    verification_defaults = {
        'accuracy_threshold': 0.7,
        'ident_threshold': 0.1,
    }
    for key, default_val in verification_defaults.items():
        if key not in resolved['verification']:
            resolved['verification'][key] = default_val

    return resolved
