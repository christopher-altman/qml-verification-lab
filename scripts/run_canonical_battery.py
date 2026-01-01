#!/usr/bin/env python3
"""Run the canonical verification battery.

This battery demonstrates the core verification gap: where predictive accuracy
remains high but epistemic metrics (identifiability, Fisher information) degrade.

Outputs:
- figures/hero_light.png
- figures/hero_dark.png
- results/summary.csv
"""

import sys
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from qvl.config import load_config
from qvl.runner import run_sweep


DARK_BG = '#121826'
LIGHT_BG = '#FFFFFF'


def create_canonical_hero_plot(
    results,
    output_path,
    dark_mode=True,
    dpi=150
):
    """Create canonical hero plot showing error vs noise × measurement noise.

    Args:
        results: List of result dictionaries
        output_path: Path to save figure
        dark_mode: Use dark background
        dpi: Resolution
    """
    if len(results) == 0:
        print("Warning: No results to plot")
        return

    # Extract unique noise values
    depol_vals = sorted(set(r['noise']['depolarizing_p'] for r in results))
    bitflip_vals = sorted(set(r['noise']['measurement_bitflip_p'] for r in results))

    # Create grids
    n_depol = len(depol_vals)
    n_bitflip = len(bitflip_vals)

    error_grid = np.zeros((n_bitflip, n_depol))
    ident_grid = np.zeros((n_bitflip, n_depol))

    # Fill grids
    for r in results:
        depol = r['noise']['depolarizing_p']
        bitflip = r['noise']['measurement_bitflip_p']

        i = bitflip_vals.index(bitflip)
        j = depol_vals.index(depol)

        # Error = 1 - accuracy (as required)
        error_grid[i, j] = 1.0 - r['metrics']['accuracy']
        ident_grid[i, j] = r['metrics']['ident_proxy']

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Set background
    if dark_mode:
        fig.patch.set_facecolor(DARK_BG)
        bg_color = DARK_BG
        text_color = 'white'
    else:
        fig.patch.set_facecolor(LIGHT_BG)
        bg_color = LIGHT_BG
        text_color = 'black'

    for ax in [ax1, ax2]:
        ax.set_facecolor(bg_color)

    # Plot error (1 - accuracy) with tight color scaling
    error_min, error_max = error_grid.min(), error_grid.max()
    im1 = ax1.imshow(
        error_grid,
        cmap='RdYlBu_r',
        aspect='auto',
        vmin=error_min,
        vmax=error_max,
        origin='lower',
        interpolation='bilinear'
    )

    ax1.set_title('Verification Error (1 − Accuracy)',
                  color=text_color, fontsize=16, pad=12, fontweight='semibold')
    ax1.set_xlabel('Feature Noise (depolarizing_p)', color=text_color, fontsize=13)
    ax1.set_ylabel('Label Noise (measurement_bitflip_p)', color=text_color, fontsize=13)

    # Set ticks
    if n_depol > 8:
        tick_indices = np.linspace(0, n_depol-1, 8, dtype=int)
    else:
        tick_indices = range(n_depol)
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels([f'{depol_vals[i]:.2f}' for i in tick_indices])

    ax1.set_yticks(range(n_bitflip))
    ax1.set_yticklabels([f'{v:.2f}' for v in bitflip_vals])

    ax1.tick_params(colors=text_color, labelsize=11)

    # Colorbar for error
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Error', color=text_color, fontsize=12)
    cbar1.ax.tick_params(colors=text_color, labelsize=10)

    # Plot identifiability with tight color scaling
    ident_min, ident_max = ident_grid.min(), ident_grid.max()
    im2 = ax2.imshow(
        ident_grid,
        cmap='plasma',
        aspect='auto',
        vmin=ident_min,
        vmax=ident_max,
        origin='lower',
        interpolation='bilinear'
    )

    ax2.set_title('Identifiability Proxy',
                  color=text_color, fontsize=16, pad=12, fontweight='semibold')
    ax2.set_xlabel('Feature Noise (depolarizing_p)', color=text_color, fontsize=13)
    ax2.set_ylabel('Label Noise (measurement_bitflip_p)', color=text_color, fontsize=13)

    # Set ticks
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels([f'{depol_vals[i]:.2f}' for i in tick_indices])

    ax2.set_yticks(range(n_bitflip))
    ax2.set_yticklabels([f'{v:.2f}' for v in bitflip_vals])

    ax2.tick_params(colors=text_color, labelsize=11)

    # Colorbar for identifiability
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Identifiability', color=text_color, fontsize=12)
    cbar2.ax.tick_params(colors=text_color, labelsize=10)

    # Set spine colors
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
            spine.set_linewidth(1.2)

    # Super title
    fig.suptitle(
        'Canonical Verification Battery: The Verification Gap',
        color=text_color,
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    # Subtitle
    fig.text(
        0.5, 0.93,
        'Accuracy remains high while identifiability collapses under noise',
        ha='center',
        color=text_color,
        fontsize=11,
        style='italic'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    print(f"Generated: {output_path}")
    print(f"  Error range: [{error_min:.4f}, {error_max:.4f}]")
    print(f"  Identifiability range: [{ident_min:.4f}, {ident_max:.4f}]")


def create_summary_csv(results, output_path):
    """Create CSV summary table.

    Args:
        results: List of result dictionaries
        output_path: Path to save CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'depolarizing_p',
            'measurement_bitflip_p',
            'accuracy',
            'error',
            'loss',
            'ident_proxy',
            'fisher_condition_number',
            'fisher_effective_rank',
            'hessian_min_abs',
            'hessian_max_abs',
            'wall_time_sec',
        ])

        # Sort by noise levels
        sorted_results = sorted(
            results,
            key=lambda r: (r['noise']['depolarizing_p'], r['noise']['measurement_bitflip_p'])
        )

        # Data rows
        for r in sorted_results:
            writer.writerow([
                r['noise']['depolarizing_p'],
                r['noise']['measurement_bitflip_p'],
                r['metrics']['accuracy'],
                1.0 - r['metrics']['accuracy'],  # error
                r['metrics']['loss'],
                r['metrics']['ident_proxy'],
                r['metrics']['fisher_condition_number'],
                r['metrics']['fisher_effective_rank'],
                r['metrics']['hessian_min_abs'],
                r['metrics']['hessian_max_abs'],
                r['timing']['wall_time_sec'],
            ])

    print(f"Generated: {output_path}")
    print(f"  Total runs: {len(results)}")


def main():
    """Run canonical verification battery."""
    print("=" * 70)
    print("Canonical Verification Battery")
    print("=" * 70)
    print()

    # Paths
    config_path = repo_root / 'batteries' / 'canonical_battery.yaml'
    figures_dir = repo_root / 'figures'
    results_dir = repo_root / 'results'

    # Load config
    print(f"Loading config: {config_path}")
    config = load_config(str(config_path))

    # Run sweep with single seed for reproducibility
    print("\nRunning verification sweep...")
    print(f"  Backend: {config['backend']}")
    print(f"  Noise sweep: {len(config['sweep']['depolarizing_p'])} × {len(config['sweep']['measurement_bitflip_p'])} = {len(config['sweep']['depolarizing_p']) * len(config['sweep']['measurement_bitflip_p'])} runs")
    print()

    summaries = run_sweep(
        config,
        output_dir=str(repo_root / 'artifacts'),
        seeds=[42],  # Single seed for canonical battery
        verbose=True
    )

    # Extract results from summaries
    results = []
    for s in summaries:
        results.append({
            'noise': s['noise'],
            'metrics': s['metrics'],
            'timing': s['timing'],
        })

    print()
    print("=" * 70)
    print("Generating outputs...")
    print("=" * 70)
    print()

    # Generate hero plots
    create_canonical_hero_plot(
        results,
        figures_dir / 'hero_dark.png',
        dark_mode=True,
        dpi=150
    )

    create_canonical_hero_plot(
        results,
        figures_dir / 'hero_light.png',
        dark_mode=False,
        dpi=150
    )

    # Generate summary CSV
    create_summary_csv(
        results,
        results_dir / 'summary.csv'
    )

    print()
    print("=" * 70)
    print("Canonical battery complete!")
    print("=" * 70)
    print()
    print("Outputs:")
    print(f"  - {figures_dir / 'hero_dark.png'}")
    print(f"  - {figures_dir / 'hero_light.png'}")
    print(f"  - {results_dir / 'summary.csv'}")
    print()
    print("These figures demonstrate the verification gap:")
    print("  High accuracy (low error) persists even as identifiability collapses.")
    print()


if __name__ == '__main__':
    main()
