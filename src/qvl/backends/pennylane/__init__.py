"""PennyLane backend for quantum circuit verification."""

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

if HAS_PENNYLANE:
    from .backend import PennyLaneBackend
    __all__ = ['PennyLaneBackend', 'HAS_PENNYLANE']
else:
    __all__ = ['HAS_PENNYLANE']
