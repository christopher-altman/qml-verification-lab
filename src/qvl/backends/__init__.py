"""Backend implementations."""

from .toy.backend import ToyBackend

BACKENDS = {
    'toy': ToyBackend,
}

try:
    from .pennylane import HAS_PENNYLANE
    if HAS_PENNYLANE:
        from .pennylane.backend import PennyLaneBackend
        BACKENDS['pennylane'] = PennyLaneBackend
except ImportError:
    pass


def get_backend(backend_name: str):
    """Get backend class by name."""
    if backend_name not in BACKENDS:
        available = ', '.join(BACKENDS.keys())
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available backends: {available}. "
            f"For 'pennylane', install with: pip install -e '.[pennylane]'"
        )
    return BACKENDS[backend_name]
