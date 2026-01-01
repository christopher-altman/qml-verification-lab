"""Tests for report generation."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

from qvl.reporting import (
    aggregate_all_results,
    create_leaderboard_csv,
    create_summary_plots,
    select_hero_point,
    create_hero_images,
    generate_report
)


def create_mock_sweep_artifacts(base_dir: Path) -> None:
    """Create mock sweep artifacts for testing.

    Args:
        base_dir: Base directory for artifacts
    """
    experiment_id = "test_sweep"

    # Create two mock runs with different noise levels
    for run_idx, (depol_p, bitflip_p) in enumerate([(0.0, 0.0), (0.1, 0.1)]):
        run_dir = base_dir / experiment_id / f"run_mock_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write config
        config = {
            "experiment_id": experiment_id,
            "backend": "toy",
            "task": "classification",
            "seed": 0,
            "noise": {
                "depolarizing_p": depol_p,
                "measurement_bitflip_p": bitflip_p,
                "amplitude_gamma": 0.0
            }
        }

        with open(run_dir / "config.resolved.json", 'w') as f:
            json.dump(config, f)

        # Write results
        results = []
        for seed in [0, 1]:
            result = {
                "metrics": {
                    "accuracy": 0.9 - depol_p,
                    "loss": 0.1 + depol_p,
                    "ident_proxy": 0.8 - 2 * depol_p,
                    "fisher_condition_number": 10.0 + 100 * depol_p,
                    "fisher_effective_rank": 5.0 - depol_p,
                },
                "noise": {
                    "depolarizing_p": depol_p,
                    "measurement_bitflip_p": bitflip_p,
                    "amplitude_gamma": 0.0
                },
                "timing": {
                    "wall_time_sec": 1.5 + run_idx * 0.5
                }
            }
            results.append(result)

        with open(run_dir / "results.jsonl", 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')


def test_aggregate_all_results():
    """Test aggregating results from multiple runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        create_mock_sweep_artifacts(base_dir)

        results = aggregate_all_results(base_dir)

        # Should find 4 results (2 runs × 2 seeds each)
        assert len(results) == 4

        # Check that metadata is populated
        for result in results:
            assert result['experiment_id'] == 'test_sweep'
            assert result['backend'] == 'toy'
            assert 'run_dir' in result
            assert 'seed' in result


def test_create_leaderboard_csv():
    """Test creating leaderboard CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        create_mock_sweep_artifacts(base_dir)

        results = aggregate_all_results(base_dir)
        output_path = Path(tmpdir) / 'leaderboard.csv'

        create_leaderboard_csv(results, output_path)

        # Check that file exists
        assert output_path.exists()

        # Check that headers are present
        with open(output_path, 'r') as f:
            header_line = f.readline().strip()
            expected_headers = [
                'experiment_id', 'backend', 'accuracy', 'loss', 'ident_proxy',
                'fisher_condition_number', 'fisher_effective_rank',
                'depolarizing_p', 'measurement_bitflip_p', 'amplitude_gamma',
                'seed', 'wall_time_sec', 'run_dir'
            ]
            for header in expected_headers:
                assert header in header_line


def test_create_leaderboard_csv_empty():
    """Test creating leaderboard CSV with empty results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'leaderboard.csv'

        create_leaderboard_csv([], output_path)

        # Should create file with headers only
        assert output_path.exists()

        with open(output_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1  # Header only
            assert 'experiment_id' in lines[0]


def test_create_summary_plots():
    """Test creating summary plots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        create_mock_sweep_artifacts(base_dir)

        results = aggregate_all_results(base_dir)
        output_dir = Path(tmpdir) / 'figures'

        create_summary_plots(results, output_dir)

        # Check that plots exist
        assert (output_dir / 'accuracy_vs_identifiability.png').exists()
        assert (output_dir / 'fisher_vs_accuracy.png').exists()
        assert (output_dir / 'identifiability_heatmap.png').exists()


def test_select_hero_point():
    """Test hero point selection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        create_mock_sweep_artifacts(base_dir)

        results = aggregate_all_results(base_dir)

        hero_result, hero_score = select_hero_point(results)

        # Should select a valid result
        assert hero_result is not None
        assert 'metrics' in hero_result
        assert isinstance(hero_score, float)


def test_select_hero_point_empty():
    """Test hero point selection with empty results."""
    hero_result, hero_score = select_hero_point([])

    # Should return empty result and zero score
    assert hero_result == {}
    assert hero_score == 0.0


def test_create_hero_images():
    """Test creating hero images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        create_mock_sweep_artifacts(base_dir)

        results = aggregate_all_results(base_dir)
        hero_result, _ = select_hero_point(results)

        output_dir = Path(tmpdir) / 'hero_output'

        create_hero_images(hero_result, output_dir)

        # Check that hero images exist
        assert (output_dir / 'hero_dark.png').exists()
        assert (output_dir / 'hero_light.png').exists()


def test_generate_report_full():
    """Test full report generation pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        create_mock_sweep_artifacts(base_dir)

        output_dir = Path(tmpdir) / 'report_output'

        # Generate full report
        generate_report(str(base_dir), str(output_dir))

        # Check that all expected outputs exist
        assert (output_dir / 'leaderboard.csv').exists()
        assert (output_dir / 'figures' / 'accuracy_vs_identifiability.png').exists()
        assert (output_dir / 'figures' / 'fisher_vs_accuracy.png').exists()
        assert (output_dir / 'figures' / 'identifiability_heatmap.png').exists()
        assert (output_dir / 'hero_dark.png').exists()
        assert (output_dir / 'hero_light.png').exists()


def test_report_cli_integration():
    """Test report command via CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        create_mock_sweep_artifacts(base_dir)

        output_dir = Path(tmpdir) / 'report_cli'

        # Run CLI command
        from qvl.cli import report_command

        class Args:
            input = str(base_dir)
            output = str(output_dir)

        args = Args()
        exit_code = report_command(args)

        # Should succeed
        assert exit_code == 0

        # Check outputs
        assert (output_dir / 'leaderboard.csv').exists()
        assert (output_dir / 'hero_dark.png').exists()
        assert (output_dir / 'hero_light.png').exists()
