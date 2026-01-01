"""Tests for PennyLane backend."""

import pytest
import sys

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

skip_if_no_pennylane = pytest.mark.skipif(
    not HAS_PENNYLANE,
    reason="PennyLane not installed. Install with: pip install -e '.[pennylane]'"
)


@skip_if_no_pennylane
def test_pennylane_backend_import():
    """Test that PennyLane backend can be imported."""
    from qvl.backends.pennylane.backend import PennyLaneBackend
    assert PennyLaneBackend is not None


@skip_if_no_pennylane
def test_pennylane_backend_instantiation():
    """Test that PennyLane backend can be instantiated."""
    from qvl.backends.pennylane.backend import PennyLaneBackend

    config = {
        'experiment_id': 'test',
        'backend': 'pennylane',
        'task': 'classification',
        'seed': 42,
        'training': {
            'n_samples': 10,
            'n_layers': 1,
            'n_epochs': 5,
            'learning_rate': 0.1,
        },
        'noise': {
            'depolarizing_p': 0.0,
            'measurement_bitflip_p': 0.0,
            'amplitude_gamma': 0.0,
        }
    }

    backend = PennyLaneBackend(config)
    assert backend.n_qubits == 2
    assert backend.n_layers == 1


@skip_if_no_pennylane
def test_pennylane_backend_run():
    """Test that PennyLane backend can run a small experiment."""
    from qvl.backends.pennylane.backend import PennyLaneBackend

    config = {
        'experiment_id': 'test',
        'backend': 'pennylane',
        'task': 'classification',
        'seed': 42,
        'training': {
            'n_samples': 20,
            'n_layers': 1,
            'n_epochs': 10,
            'learning_rate': 0.1,
        },
        'noise': {
            'depolarizing_p': 0.0,
            'measurement_bitflip_p': 0.0,
            'amplitude_gamma': 0.0,
        }
    }

    backend = PennyLaneBackend(config)
    result = backend.run()

    assert 'metrics' in result
    assert 'timing' in result
    assert 'noise' in result

    assert 'accuracy' in result['metrics']
    assert 'loss' in result['metrics']
    assert 'ident_proxy' in result['metrics']
    assert 'hessian_min_abs' in result['metrics']
    assert 'hessian_max_abs' in result['metrics']

    assert 0.0 <= result['metrics']['accuracy'] <= 1.0


@skip_if_no_pennylane
def test_pennylane_with_noise():
    """Test PennyLane backend with noise channels."""
    from qvl.backends.pennylane.backend import PennyLaneBackend

    config = {
        'experiment_id': 'test_noise',
        'backend': 'pennylane',
        'task': 'classification',
        'seed': 42,
        'training': {
            'n_samples': 20,
            'n_layers': 1,
            'n_epochs': 10,
            'learning_rate': 0.1,
        },
        'noise': {
            'depolarizing_p': 0.1,
            'measurement_bitflip_p': 0.05,
            'amplitude_gamma': 0.05,
        }
    }

    backend = PennyLaneBackend(config)
    result = backend.run()

    assert result['noise']['depolarizing_p'] == 0.1
    assert result['noise']['measurement_bitflip_p'] == 0.05
    assert result['noise']['amplitude_gamma'] == 0.05


@skip_if_no_pennylane
def test_pennylane_cli_smoke():
    """Test that CLI can run with PennyLane backend."""
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(__file__).parent.parent / 'examples' / 'vqc_smoke.yaml'

        if not config_path.exists():
            pytest.skip("vqc_smoke.yaml not found")

        result = subprocess.run(
            [
                sys.executable, '-m', 'qvl', 'run',
                '--config', str(config_path),
                '--output-dir', tmpdir,
                '--seed', '42',
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        output_path = Path(tmpdir)
        experiment_dirs = list(output_path.glob('*/run_*'))
        assert len(experiment_dirs) > 0

        run_dir = experiment_dirs[0]
        assert (run_dir / 'summary.json').exists()
