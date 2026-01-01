"""CLI smoke tests."""

import subprocess
import sys
import tempfile
from pathlib import Path


def test_cli_help():
    """Test that --help works."""
    result = subprocess.run(
        [sys.executable, '-m', 'qvl', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'Quantum Machine Learning Verification Laboratory' in result.stdout


def test_cli_run_smoke():
    """Test that run command produces expected artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(__file__).parent.parent / 'examples' / 'toy_smoke.yaml'

        result = subprocess.run(
            [
                sys.executable, '-m', 'qvl', 'run',
                '--config', str(config_path),
                '--output-dir', tmpdir,
                '--seed', '42',
            ],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0

        output_path = Path(tmpdir)
        experiment_dirs = list(output_path.glob('*/run_*'))
        assert len(experiment_dirs) > 0, f"No run directories found in {tmpdir}"

        run_dir = experiment_dirs[0]

        assert (run_dir / 'summary.json').exists(), "summary.json not found"
        assert (run_dir / 'config.resolved.json').exists(), "config.resolved.json not found"
        assert (run_dir / 'results.jsonl').exists(), "results.jsonl not found"
        assert (run_dir / 'env.json').exists(), "env.json not found"

        figures_dir = run_dir / 'figures'
        assert figures_dir.exists(), "figures directory not found"
        assert (figures_dir / 'hero_identifiability_dark.png').exists(), "dark hero plot not found"
        assert (figures_dir / 'hero_identifiability_light.png').exists(), "light hero plot not found"


def test_cli_report_stub():
    """Test that report command exists (stub)."""
    result = subprocess.run(
        [sys.executable, '-m', 'qvl', 'report'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'not yet implemented' in result.stdout
