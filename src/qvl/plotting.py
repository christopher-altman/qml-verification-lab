"""Plotting utilities for hero figures."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List


DARK_BG = '#121826'
LIGHT_BG = '#FFFFFF'


def create_hero_identifiability_plot(
    results: List[Dict[str, Any]],
    output_path: Path,
    dark_mode: bool = True,
    transparent: bool = False
) -> None:
    """Create hero identifiability plot.

    Args:
        results: List of result dictionaries
        output_path: Path to save figure
        dark_mode: Use dark background
        transparent: Use transparent background
    """
    if len(results) == 0:
        results = [{'noise': {'depolarizing_p': 0.0, 'measurement_bitflip_p': 0.0},
                   'metrics': {'accuracy': 0.0, 'ident_proxy': 0.0}}]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if transparent:
        fig.patch.set_facecolor('none')
        bg_color = 'none'
    elif dark_mode:
        fig.patch.set_facecolor(DARK_BG)
        bg_color = DARK_BG
    else:
        fig.patch.set_facecolor(LIGHT_BG)
        bg_color = LIGHT_BG

    text_color = 'white' if dark_mode else 'black'

    depol_vals = sorted(set(r['noise'].get('depolarizing_p', 0.0) for r in results))
    bitflip_vals = sorted(set(r['noise'].get('measurement_bitflip_p', 0.0) for r in results))

    if len(depol_vals) == 0:
        depol_vals = [0.0]
    if len(bitflip_vals) == 0:
        bitflip_vals = [0.0]

    acc_grid = np.zeros((len(bitflip_vals), len(depol_vals)))
    ident_grid = np.zeros((len(bitflip_vals), len(depol_vals)))

    for r in results:
        depol = r['noise'].get('depolarizing_p', 0.0)
        bitflip = r['noise'].get('measurement_bitflip_p', 0.0)

        if depol in depol_vals and bitflip in bitflip_vals:
            i = bitflip_vals.index(bitflip)
            j = depol_vals.index(depol)
            acc_grid[i, j] = r['metrics'].get('accuracy', 0.0)
            ident_grid[i, j] = r['metrics'].get('ident_proxy', 0.0)

    for ax in axes:
        if transparent:
            ax.set_facecolor('none')
            ax.patch.set_alpha(0.0)
        elif dark_mode:
            ax.set_facecolor(DARK_BG)
        else:
            ax.set_facecolor(LIGHT_BG)

    im0 = axes[0].imshow(acc_grid, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('Accuracy', color=text_color, fontsize=14)
    axes[0].set_xlabel('Depolarizing p', color=text_color)
    axes[0].set_ylabel('Measurement Bitflip p', color=text_color)
    axes[0].tick_params(colors=text_color)

    if len(depol_vals) > 1:
        axes[0].set_xticks(range(len(depol_vals)))
        axes[0].set_xticklabels([f'{v:.2f}' for v in depol_vals])
    if len(bitflip_vals) > 1:
        axes[0].set_yticks(range(len(bitflip_vals)))
        axes[0].set_yticklabels([f'{v:.2f}' for v in bitflip_vals])

    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.ax.tick_params(colors=text_color)

    im1 = axes[1].imshow(ident_grid, cmap='plasma', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Identifiability Proxy', color=text_color, fontsize=14)
    axes[1].set_xlabel('Depolarizing p', color=text_color)
    axes[1].set_ylabel('Measurement Bitflip p', color=text_color)
    axes[1].tick_params(colors=text_color)

    if len(depol_vals) > 1:
        axes[1].set_xticks(range(len(depol_vals)))
        axes[1].set_xticklabels([f'{v:.2f}' for v in depol_vals])
    if len(bitflip_vals) > 1:
        axes[1].set_yticks(range(len(bitflip_vals)))
        axes[1].set_yticklabels([f'{v:.2f}' for v in bitflip_vals])

    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.ax.tick_params(colors=text_color)

    for spine in axes[0].spines.values():
        spine.set_edgecolor(text_color)
    for spine in axes[1].spines.values():
        spine.set_edgecolor(text_color)

    fig.suptitle(
        'Accuracy vs Identifiability: Noise Can Degrade Identifiability Faster Than Accuracy',
        color=text_color,
        fontsize=11,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def generate_all_hero_plots(results: List[Dict[str, Any]], figures_dir: Path) -> None:
    """Generate all hero plot variants.

    Args:
        results: List of result dictionaries
        figures_dir: Directory to save figures
    """
    create_hero_identifiability_plot(
        results,
        figures_dir / 'hero_identifiability_dark.png',
        dark_mode=True,
        transparent=False
    )

    create_hero_identifiability_plot(
        results,
        figures_dir / 'hero_identifiability_light.png',
        dark_mode=False,
        transparent=False
    )

    create_hero_identifiability_plot(
        results,
        figures_dir / 'hero_identifiability_dark_transparent.png',
        dark_mode=True,
        transparent=True
    )

    create_hero_identifiability_plot(
        results,
        figures_dir / 'hero_identifiability_light_transparent.png',
        dark_mode=False,
        transparent=True
    )
