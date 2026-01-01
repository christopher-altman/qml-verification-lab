"""Report generation from sweep artifacts."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def discover_results_files(input_path: Path) -> List[Path]:
    """Discover all results.jsonl files under input path.

    Args:
        input_path: Directory to search

    Returns:
        List of paths to results.jsonl files
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    results_files = []

    # Check if input_path itself contains results.jsonl
    if (input_path / "results.jsonl").exists():
        results_files.append(input_path / "results.jsonl")

    # Recursively search for results.jsonl files
    for jsonl_file in input_path.rglob("results.jsonl"):
        if jsonl_file not in results_files:
            results_files.append(jsonl_file)

    return sorted(results_files)


def load_results_from_jsonl(jsonl_path: Path, run_dir: Path) -> List[Dict[str, Any]]:
    """Load results from a single JSONL file.

    Args:
        jsonl_path: Path to results.jsonl file
        run_dir: Run directory (for metadata)

    Returns:
        List of result dictionaries with added metadata
    """
    results = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                result['run_dir'] = str(run_dir)
                results.append(result)

    return results


def load_config_from_run(run_dir: Path) -> Dict[str, Any]:
    """Load config from run directory.

    Args:
        run_dir: Run directory

    Returns:
        Configuration dictionary
    """
    config_path = run_dir / "config.resolved.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def aggregate_all_results(input_path: Path) -> List[Dict[str, Any]]:
    """Aggregate all results from input directory.

    Args:
        input_path: Directory containing runs

    Returns:
        List of aggregated result dictionaries
    """
    results_files = discover_results_files(input_path)

    if not results_files:
        raise ValueError(f"No results.jsonl files found under {input_path}")

    all_results = []

    for jsonl_path in results_files:
        run_dir = jsonl_path.parent
        config = load_config_from_run(run_dir)

        # Load results from this run
        results = load_results_from_jsonl(jsonl_path, run_dir)

        # Enrich with config metadata
        for result in results:
            result['experiment_id'] = config.get('experiment_id', 'unknown')
            result['backend'] = config.get('backend', 'unknown')
            result['seed'] = config.get('seed', 0)

        all_results.extend(results)

    return all_results


def create_leaderboard_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Create leaderboard CSV from results.

    Args:
        results: List of result dictionaries
        output_path: Path to write CSV
    """
    if not results:
        # Write empty CSV with headers
        headers = [
            'experiment_id', 'backend', 'accuracy', 'loss', 'ident_proxy',
            'fisher_condition_number', 'fisher_effective_rank',
            'depolarizing_p', 'measurement_bitflip_p', 'amplitude_gamma',
            'seed', 'wall_time_sec', 'run_dir'
        ]
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        return

    # Sort results deterministically for reproducibility
    sorted_results = sorted(results, key=lambda r: (
        r.get('experiment_id', ''),
        r.get('backend', ''),
        r.get('noise', {}).get('depolarizing_p', 0.0),
        r.get('noise', {}).get('measurement_bitflip_p', 0.0),
        r.get('noise', {}).get('amplitude_gamma', 0.0),
        r.get('seed', 0)
    ))

    # Define stable column order
    headers = [
        'experiment_id', 'backend', 'accuracy', 'loss', 'ident_proxy',
        'fisher_condition_number', 'fisher_effective_rank',
        'depolarizing_p', 'measurement_bitflip_p', 'amplitude_gamma',
        'seed', 'wall_time_sec', 'run_dir'
    ]

    rows = []
    for result in sorted_results:
        metrics = result.get('metrics', {})
        noise = result.get('noise', {})
        timing = result.get('timing', {})

        row = {
            'experiment_id': result.get('experiment_id', 'unknown'),
            'backend': result.get('backend', 'unknown'),
            'accuracy': metrics.get('accuracy', 0.0),
            'loss': metrics.get('loss', 0.0),
            'ident_proxy': metrics.get('ident_proxy', 0.0),
            'fisher_condition_number': metrics.get('fisher_condition_number', 0.0),
            'fisher_effective_rank': metrics.get('fisher_effective_rank', 0.0),
            'depolarizing_p': noise.get('depolarizing_p', 0.0),
            'measurement_bitflip_p': noise.get('measurement_bitflip_p', 0.0),
            'amplitude_gamma': noise.get('amplitude_gamma', 0.0),
            'seed': result.get('seed', 0),
            'wall_time_sec': timing.get('wall_time_sec', 0.0),
            'run_dir': result.get('run_dir', '')
        }
        rows.append(row)

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def create_summary_plots(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Create canonical summary plots.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    if not results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics
    accuracies = []
    ident_proxies = []
    fisher_conds = []
    depol_ps = []
    bitflip_ps = []

    for r in results:
        metrics = r.get('metrics', {})
        noise = r.get('noise', {})

        accuracies.append(metrics.get('accuracy', 0.0))
        ident_proxies.append(metrics.get('ident_proxy', 0.0))
        fisher_conds.append(metrics.get('fisher_condition_number', 1.0))
        depol_ps.append(noise.get('depolarizing_p', 0.0))
        bitflip_ps.append(noise.get('measurement_bitflip_p', 0.0))

    # Plot 1: Accuracy vs Identifiability Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(ident_proxies, accuracies, c=fisher_conds,
                        cmap='viridis', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Identifiability Proxy', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Identifiability (colored by Fisher Condition Number)', fontsize=13)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fisher Condition Number', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_identifiability.png', dpi=150)
    plt.close()

    # Plot 2: Fisher Condition Number vs Accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(accuracies, fisher_conds, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Fisher Condition Number', fontsize=12)
    ax.set_title('Fisher Condition Number vs Accuracy', fontsize=13)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'fisher_vs_accuracy.png', dpi=150)
    plt.close()

    # Plot 3: Identifiability Heatmap (if grid-like data detected)
    unique_depol = sorted(set(depol_ps))
    unique_bitflip = sorted(set(bitflip_ps))

    if len(unique_depol) > 1 and len(unique_bitflip) > 1:
        # Create heatmap grid
        ident_grid = np.full((len(unique_bitflip), len(unique_depol)), np.nan)

        for i, bf in enumerate(unique_bitflip):
            for j, dp in enumerate(unique_depol):
                # Find matching results and average
                matching = [
                    r.get('metrics', {}).get('ident_proxy', 0.0)
                    for r in results
                    if abs(r.get('noise', {}).get('depolarizing_p', 0.0) - dp) < 1e-9
                    and abs(r.get('noise', {}).get('measurement_bitflip_p', 0.0) - bf) < 1e-9
                ]
                if matching:
                    ident_grid[i, j] = np.mean(matching)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(ident_grid, cmap='plasma', aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel('Depolarizing p', fontsize=12)
        ax.set_ylabel('Measurement Bitflip p', fontsize=12)
        ax.set_title('Identifiability Proxy Heatmap', fontsize=13)

        ax.set_xticks(range(len(unique_depol)))
        ax.set_xticklabels([f'{v:.2f}' for v in unique_depol])
        ax.set_yticks(range(len(unique_bitflip)))
        ax.set_yticklabels([f'{v:.2f}' for v in unique_bitflip])

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Identifiability Proxy', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / 'identifiability_heatmap.png', dpi=150)
        plt.close()


def select_hero_point(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
    """Select hero point for storytelling.

    Selection heuristic:
    - High accuracy is good, but penalize low ident_proxy / high fisher_condition_number
    - Prefer points where tension is visible (accuracy high but ident low)

    Score = accuracy - 0.5 * (1 - ident_proxy) - 0.1 * log10(fisher_cond)

    Args:
        results: List of result dictionaries

    Returns:
        Tuple of (hero_result, hero_score)
    """
    if not results:
        return {}, 0.0

    best_result = None
    best_score = -float('inf')

    for result in results:
        metrics = result.get('metrics', {})

        accuracy = metrics.get('accuracy', 0.0)
        ident_proxy = metrics.get('ident_proxy', 0.0)
        fisher_cond = max(metrics.get('fisher_condition_number', 1.0), 1.0)

        # Scoring function: reward high accuracy, penalize low identifiability and high condition number
        # But prefer points showing the "tension" (high acc, low ident)
        score = accuracy - 0.5 * (1 - ident_proxy) - 0.1 * np.log10(fisher_cond)

        # Bonus for "storytelling tension": high accuracy (>0.7) but low identifiability (<0.3)
        if accuracy > 0.7 and ident_proxy < 0.3:
            score += 0.2

        if score > best_score:
            best_score = score
            best_result = result

    return best_result, best_score


def create_hero_images(hero_result: Dict[str, Any], output_dir: Path) -> None:
    """Create hero images (dark and light variants) from hero point.

    Args:
        hero_result: Selected hero result dictionary
        output_dir: Directory to save images
    """
    if not hero_result:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = hero_result.get('metrics', {})
    noise = hero_result.get('noise', {})

    accuracy = metrics.get('accuracy', 0.0)
    ident_proxy = metrics.get('ident_proxy', 0.0)
    fisher_cond = metrics.get('fisher_condition_number', 1.0)
    depol_p = noise.get('depolarizing_p', 0.0)
    bitflip_p = noise.get('measurement_bitflip_p', 0.0)

    # Create hero visualization for both dark and light modes
    for dark_mode in [True, False]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        bg_color = '#121826' if dark_mode else '#FFFFFF'
        text_color = 'white' if dark_mode else 'black'

        fig.patch.set_facecolor(bg_color)
        ax1.set_facecolor(bg_color)
        ax2.set_facecolor(bg_color)

        # Left plot: Accuracy vs Identifiability
        ax1.scatter([ident_proxy], [accuracy], s=300, c='red', marker='*',
                   edgecolors='yellow', linewidth=2, zorder=5, label='Hero Point')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Identifiability Proxy', color=text_color, fontsize=12)
        ax1.set_ylabel('Accuracy', color=text_color, fontsize=12)
        ax1.set_title('Hero Point: Accuracy vs Identifiability', color=text_color, fontsize=13)
        ax1.grid(True, alpha=0.3, color=text_color)
        ax1.tick_params(colors=text_color)
        for spine in ax1.spines.values():
            spine.set_edgecolor(text_color)
        ax1.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)

        # Right plot: Metric summary bars
        metric_names = ['Accuracy', 'Ident Proxy', 'Fisher Cond\n(normalized)']
        metric_values = [
            accuracy,
            ident_proxy,
            min(fisher_cond / 1000.0, 1.0)  # Normalize to [0, 1] range
        ]
        colors_bars = ['#3498db', '#e74c3c', '#f39c12']

        bars = ax2.bar(metric_names, metric_values, color=colors_bars, edgecolor=text_color, linewidth=1.5)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Metric Value', color=text_color, fontsize=12)
        ax2.set_title('Hero Point Metrics', color=text_color, fontsize=13)
        ax2.tick_params(colors=text_color)
        ax2.grid(True, axis='y', alpha=0.3, color=text_color)
        for spine in ax2.spines.values():
            spine.set_edgecolor(text_color)

        # Add value labels on bars
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', color=text_color, fontsize=10)

        # Subtitle with noise parameters
        subtitle = (f'Noise: depol_p={depol_p:.3f}, bitflip_p={bitflip_p:.3f} | '
                   f'Fisher Cond={fisher_cond:.1f}')
        fig.suptitle(subtitle, color=text_color, fontsize=10, y=0.96)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        suffix = 'dark' if dark_mode else 'light'
        plt.savefig(output_dir / f'hero_{suffix}.png', dpi=150, facecolor=bg_color)
        plt.close()


def generate_markdown_summary(
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_path: Path,
    figures_dir: Path
) -> None:
    """Generate Markdown summary report.

    Args:
        results: List of result dictionaries
        config: Experiment configuration (from first run)
        output_path: Path to write summary.md
        figures_dir: Directory containing figure files (for relative paths)
    """
    lines = []

    # Header
    lines.append(f"# Experiment Report: {config.get('experiment_id', 'unknown')}")
    lines.append("")
    lines.append(f"**Backend:** {config.get('backend', 'unknown')}")
    lines.append(f"**Task:** {config.get('task', 'unknown')}")
    lines.append(f"**Total Runs:** {len(results)}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2))
    lines.append("```")
    lines.append("")

    # Key Metrics Summary
    lines.append("## Key Metrics Summary")
    lines.append("")

    if results:
        accuracies = [r.get('metrics', {}).get('accuracy', 0.0) for r in results]
        ident_proxies = [r.get('metrics', {}).get('ident_proxy', 0.0) for r in results]
        fisher_conds = [r.get('metrics', {}).get('fisher_condition_number', 1.0) for r in results]

        lines.append(f"| Metric | Min | Max | Mean | Median |")
        lines.append(f"|--------|-----|-----|------|--------|")
        lines.append(f"| Accuracy | {min(accuracies):.4f} | {max(accuracies):.4f} | {np.mean(accuracies):.4f} | {np.median(accuracies):.4f} |")
        lines.append(f"| Identifiability Proxy | {min(ident_proxies):.4f} | {max(ident_proxies):.4f} | {np.mean(ident_proxies):.4f} | {np.median(ident_proxies):.4f} |")
        lines.append(f"| Fisher Condition Number | {min(fisher_conds):.2f} | {max(fisher_conds):.2f} | {np.mean(fisher_conds):.2f} | {np.median(fisher_conds):.2f} |")
    else:
        lines.append("No results available.")

    lines.append("")

    # Plots section
    lines.append("## Visualizations")
    lines.append("")

    # Check for hero images
    hero_dark = figures_dir.parent / "hero_dark.png"
    hero_light = figures_dir.parent / "hero_light.png"
    if hero_dark.exists():
        lines.append(f"### Hero Point")
        lines.append("")
        lines.append(f"![Hero Point Dark](hero_dark.png)")
        lines.append("")

    # Check for summary plots
    acc_vs_ident = figures_dir / "accuracy_vs_identifiability.png"
    if acc_vs_ident.exists():
        lines.append(f"### Accuracy vs Identifiability")
        lines.append("")
        lines.append(f"![Accuracy vs Identifiability](figures/accuracy_vs_identifiability.png)")
        lines.append("")

    fisher_vs_acc = figures_dir / "fisher_vs_accuracy.png"
    if fisher_vs_acc.exists():
        lines.append(f"### Fisher Condition Number vs Accuracy")
        lines.append("")
        lines.append(f"![Fisher vs Accuracy](figures/fisher_vs_accuracy.png)")
        lines.append("")

    ident_heatmap = figures_dir / "identifiability_heatmap.png"
    if ident_heatmap.exists():
        lines.append(f"### Identifiability Heatmap")
        lines.append("")
        lines.append(f"![Identifiability Heatmap](figures/identifiability_heatmap.png)")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by QVL*")
    lines.append("")

    # Write markdown file
    output_path.write_text("\n".join(lines))


def generate_report(input_dir: str, output_dir: str) -> None:
    """Generate full report from sweep artifacts.

    Args:
        input_dir: Directory containing sweep runs
        output_dir: Directory to write report outputs
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Discovering results under: {input_path}")
    results = aggregate_all_results(input_path)
    print(f"Found {len(results)} result(s)")

    # Load config from first run for summary
    results_files = discover_results_files(input_path)
    config = {}
    if results_files:
        config = load_config_from_run(results_files[0].parent)

    # Create leaderboard
    leaderboard_path = output_path / 'leaderboard.csv'
    print(f"Writing leaderboard to: {leaderboard_path}")
    create_leaderboard_csv(results, leaderboard_path)

    # Create summary plots
    figures_dir = output_path / 'figures'
    print(f"Generating summary plots in: {figures_dir}")
    create_summary_plots(results, figures_dir)

    # Select hero point and create hero images
    print("Selecting hero point...")
    hero_result, hero_score = select_hero_point(results)

    if hero_result:
        print(f"Hero point selected with score: {hero_score:.3f}")
        print(f"  Accuracy: {hero_result.get('metrics', {}).get('accuracy', 0.0):.3f}")
        print(f"  Ident Proxy: {hero_result.get('metrics', {}).get('ident_proxy', 0.0):.3f}")
        print(f"  Fisher Cond: {hero_result.get('metrics', {}).get('fisher_condition_number', 1.0):.1f}")

        print("Creating hero images...")
        create_hero_images(hero_result, output_path)
    else:
        print("No valid hero point found")

    # Generate Markdown summary
    summary_path = output_path / 'summary.md'
    print(f"Writing Markdown summary to: {summary_path}")
    generate_markdown_summary(results, config, summary_path, figures_dir)

    print(f"\nReport generation complete. Outputs in: {output_path}")
