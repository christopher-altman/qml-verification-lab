"""Backend implementations."""

from .toy.backend import ToyBackend

BACKENDS = {
    'toy': ToyBackend,
}


def get_backend(backend_name: str):
    """Get backend class by name."""
    if backend_name not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKENDS[backend_name]
