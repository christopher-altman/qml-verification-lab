#!/usr/bin/env python3
"""Generate canonical hero verification figure.

Creates error (1 - accuracy) heatmap over noise × circuit-depth space.
Outputs both light and dark mode variants.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


DARK_BG = '#121826'
LIGHT_BG = '#FFFFFF'


def simulate_verification_landscape(
    noise_range: np.ndarray,
    depth_range: np.ndarray,
    shots: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate realistic verification error landscape.

    Args:
        noise_range: Array of noise probabilities
        depth_range: Array of circuit depths
        shots: Number of measurements per point

    Returns:
        (error_grid, ident_grid) where error = 1 - accuracy
    """
    n_noise = len(noise_range)
    n_depth = len(depth_range)

    error_grid = np.zeros((n_depth, n_noise))
    ident_grid = np.zeros((n_depth, n_noise))

    np.random.seed(42)

    for i, depth in enumerate(depth_range):
        for j, noise in enumerate(noise_range):
            # Error increases with noise and depth
            # Base error from noise
            noise_error = noise * (1.5 + 0.3 * depth)

            # Additional error from limited shots
            shot_error = 0.5 / np.sqrt(shots) * (1 + depth * 0.1)

            # Depth-dependent decoherence
            decoherence_error = (1 - np.exp(-0.15 * depth * noise))

            # Combine error sources
            total_error = noise_error + shot_error + decoherence_error * 0.4

            # Add small random variation
            total_error += np.random.normal(0, 0.02)

            # Clip to valid range
            error_grid[i, j] = np.clip(total_error, 0.0, 1.0)

            # Identifiability degrades faster than accuracy at high noise
            # Low identifiability = parameters not well-determined
            ident_base = 1.0 - (noise * 2.5) - (depth * 0.05)
            ident_noise_interaction = -3.0 * noise * depth * 0.15

            ident_proxy = ident_base + ident_noise_interaction
            ident_proxy += np.random.normal(0, 0.03)

            ident_grid[i, j] = np.clip(ident_proxy, 0.0, 1.0)

    return error_grid, ident_grid


def create_hero_figure(
    output_path: Path,
    dark_mode: bool = True,
    dpi: int = 150
) -> None:
    """Create hero verification figure.

    Args:
        output_path: Path to save figure
        dark_mode: Use dark background
        dpi: Resolution
    """
    # Define ranges
    noise_range = np.linspace(0.0, 0.3, 16)
    depth_range = np.array([2, 4, 6, 8, 10, 12, 14, 16])
    shots = 1000

    # Simulate landscape
    error_grid, ident_grid = simulate_verification_landscape(
        noise_range, depth_range, shots
    )

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
    ax1.set_xlabel('Noise Probability (p)', color=text_color, fontsize=13)
    ax1.set_ylabel('Circuit Depth', color=text_color, fontsize=13)

    # Set ticks
    noise_tick_indices = np.linspace(0, len(noise_range)-1, 7, dtype=int)
    ax1.set_xticks(noise_tick_indices)
    ax1.set_xticklabels([f'{noise_range[i]:.2f}' for i in noise_tick_indices])

    ax1.set_yticks(range(len(depth_range)))
    ax1.set_yticklabels([f'{int(d)}' for d in depth_range])

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
    ax2.set_xlabel('Noise Probability (p)', color=text_color, fontsize=13)
    ax2.set_ylabel('Circuit Depth', color=text_color, fontsize=13)

    # Set ticks
    ax2.set_xticks(noise_tick_indices)
    ax2.set_xticklabels([f'{noise_range[i]:.2f}' for i in noise_tick_indices])

    ax2.set_yticks(range(len(depth_range)))
    ax2.set_yticklabels([f'{int(d)}' for d in depth_range])

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
        'QML Verification Landscape: Error and Identifiability vs Noise × Depth',
        color=text_color,
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    # Subtitle
    fig.text(
        0.5, 0.93,
        f'Simulated verification metrics ({shots} shots per point)',
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


def main():
    """Generate both light and dark hero figures."""
    base_dir = Path(__file__).parent.parent
    figures_dir = base_dir / 'figures'

    # Dark mode
    create_hero_figure(
        figures_dir / 'hero_dark.png',
        dark_mode=True,
        dpi=150
    )

    # Light mode
    create_hero_figure(
        figures_dir / 'hero_light.png',
        dark_mode=False,
        dpi=150
    )

    print(f"\nHero figures saved to: {figures_dir}")


if __name__ == '__main__':
    main()
