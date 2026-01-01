"""Experiment runner and orchestration."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .artifacts import ArtifactWriter, get_env_info, try_get_git_info
from .backends import get_backend
from .config import compute_config_hash, resolve_config
from .plotting import generate_all_hero_plots


def run_single_experiment(
    config: Dict[str, Any],
    output_dir: str,
    seed: int,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run a single experiment.

    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        seed: Random seed
        verbose: Print progress

    Returns:
        Summary dictionary
    """
    resolved_config = resolve_config(config, seed)

    experiment_id = resolved_config['experiment_id']
    config_hash = compute_config_hash(resolved_config)

    writer = ArtifactWriter(output_dir, experiment_id, config_hash, seed)

    writer.write_config(resolved_config)
    writer.write_env(get_env_info())

    git_info = try_get_git_info()
    if git_info:
        writer.write_git_info(git_info)

    backend_name = resolved_config['backend']
    backend_cls = get_backend(backend_name)
    backend = backend_cls(resolved_config)

    if verbose:
        print(f"Running experiment: {experiment_id}")
        print(f"  Backend: {backend_name}")
        print(f"  Seed: {seed}")
        print(f"  Noise: {resolved_config['noise']}")

    result = backend.run()

    timestamp_utc = datetime.now(timezone.utc).isoformat()

    verification_config = resolved_config.get('verification', {})
    verification_warning = check_verification_warning(
        result['metrics'],
        accuracy_threshold=verification_config.get('accuracy_threshold', 0.7),
        ident_threshold=verification_config.get('ident_threshold', 0.1)
    )

    summary = {
        'experiment_id': experiment_id,
        'backend': backend_name,
        'task': resolved_config['task'],
        'timestamp_utc': timestamp_utc,
        'config_hash': config_hash,
        'seed': seed,
        'metrics': result['metrics'],
        'noise': result['noise'],
        'timing': result['timing'],
        'notes': {
            'verification_warning': verification_warning['warning'],
            'warning_reason': verification_warning['reason'],
        },
    }

    writer.write_summary(summary)

    result_line = {
        'seed': seed,
        'noise': result['noise'],
        'metrics': result['metrics'],
        'timing': result['timing'],
    }
    writer.append_result(result_line)

    generate_all_hero_plots([result_line], writer.figures_dir)

    leaderboard_path = writer.get_table_path('leaderboard.csv')
    leaderboard_path.write_text('experiment,accuracy,identifiability\n')

    if verbose:
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  Identifiability Proxy: {result['metrics']['ident_proxy']:.4f}")
        if verification_warning['warning']:
            print(f"  WARNING: {verification_warning['reason']}")
        print(f"  Artifacts saved to: {writer.run_dir}")

    return summary


def run_sweep(
    config: Dict[str, Any],
    output_dir: str,
    seeds: List[int],
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run a parameter sweep.

    Args:
        config: Configuration dictionary with sweep parameters
        output_dir: Output directory path
        seeds: List of random seeds
        verbose: Print progress

    Returns:
        List of summary dictionaries
    """
    experiment_id = config['experiment_id']
    sweep_config = config.get('sweep', {})

    sweep_points = generate_sweep_points(sweep_config)

    if verbose:
        print(f"Running sweep: {experiment_id}")
        print(f"  Sweep points: {len(sweep_points)}")
        print(f"  Seeds: {seeds}")
        print(f"  Total runs: {len(sweep_points) * len(seeds)}")

    total_runs = len(sweep_points) * len(seeds)
    run_idx = 0

    all_results = []
    all_summaries = []

    if HAS_TQDM and verbose:
        pbar = tqdm(total=total_runs, desc="Sweep progress")
    else:
        pbar = None

    for point_idx, point in enumerate(sweep_points):
        for seed in seeds:
            run_config = config.copy()

            if 'noise' not in run_config:
                run_config['noise'] = {}
            run_config['noise'].update(point)

            if verbose and not HAS_TQDM:
                print(f"\nRun {run_idx + 1}/{total_runs}")
                print(f"  Point {point_idx + 1}/{len(sweep_points)}: {point}")
                print(f"  Seed: {seed}")

            summary = run_single_experiment(run_config, output_dir, seed, verbose=False)
            all_summaries.append(summary)

            result_line = {
                'seed': seed,
                'noise': summary['noise'],
                'metrics': summary['metrics'],
                'timing': summary['timing'],
            }
            all_results.append(result_line)

            run_idx += 1

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    if len(all_results) > 0:
        config_hash = compute_config_hash(config)
        # For sweeps, use first seed for directory naming
        sweep_seed = seeds[0] if seeds else 0
        writer = ArtifactWriter(output_dir, experiment_id, config_hash, sweep_seed)

        for result in all_results:
            writer.append_result(result)

        generate_all_hero_plots(all_results, writer.figures_dir)

        aggregate_summary = {
            'experiment_id': experiment_id,
            'sweep_points': len(sweep_points),
            'seeds': seeds,
            'total_runs': total_runs,
            'results': all_summaries,
        }
        writer.write_summary(aggregate_summary)

        if verbose:
            print(f"\nSweep complete!")
            print(f"  Artifacts saved to: {writer.run_dir}")

    return all_summaries


def generate_sweep_points(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate sweep points from configuration.

    Args:
        sweep_config: Sweep configuration

    Returns:
        List of parameter dictionaries
    """
    if not sweep_config:
        return [{}]

    import itertools

    param_names = []
    param_values = []

    for param_name, values in sweep_config.items():
        param_names.append(param_name)
        param_values.append(values)

    points = []
    for combination in itertools.product(*param_values):
        point = dict(zip(param_names, combination))
        points.append(point)

    return points


def check_verification_warning(
    metrics: Dict[str, float],
    accuracy_threshold: float = 0.7,
    ident_threshold: float = 0.1
) -> Dict[str, Any]:
    """Check if verification warning should be raised.

    Args:
        metrics: Metrics dictionary
        accuracy_threshold: Minimum accuracy to trigger warning
        ident_threshold: Maximum identifiability to trigger warning

    Returns:
        Dictionary with 'warning' (bool) and 'reason' (str)
    """
    accuracy = metrics.get('accuracy', 0.0)
    ident_proxy = metrics.get('ident_proxy', 0.0)

    if accuracy >= accuracy_threshold and ident_proxy <= ident_threshold:
        return {
            'warning': True,
            'reason': f'High accuracy ({accuracy:.3f}) but low identifiability ({ident_proxy:.3f}) - potential overfitting or noise dominance',
        }

    return {
        'warning': False,
        'reason': '',
    }
