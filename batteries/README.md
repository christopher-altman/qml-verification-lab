# Canonical Verification Battery

The canonical battery is a single-command demonstration of the verification gap that serves as the conceptual anchor of QVL.

## What it demonstrates

**The verification gap**: Predictive accuracy remains high even as epistemic metrics (identifiability, Fisher information) collapse under noise.

- **Left panel**: Verification error (1 − accuracy) stays low across noise regimes
- **Right panel**: Identifiability proxy degrades rapidly with noise
- **The gap**: Models appear to "work" but parameters are not identifiable

## Running the battery

Single command:

```bash
python -m qvl battery
```

Or directly:

```bash
python scripts/run_canonical_battery.py
```

## Outputs

- `figures/hero_light.png` - Light background hero plot
- `figures/hero_dark.png` - Dark background hero plot (for presentations/README)
- `results/summary.csv` - Full tabular results

## Configuration

The battery uses `batteries/canonical_battery.yaml`:

- **Backend**: `toy` (logistic regression, fast, reproducible)
- **Noise ranges**:
  - Feature noise (depolarizing_p): [0.0, 0.3] in 16 steps
  - Label noise (measurement_bitflip_p): [0.0, 0.3] in 7 steps
- **Total runs**: 112 (16 × 7)
- **Seed**: 42 (fixed for reproducibility)

## Plot features

- **Error = 1 − accuracy** (not accuracy directly)
- **Tight color scaling** (no flat fields)
- **2D structure**: noise × measurement noise creates visual gradients
- **Bilinear interpolation** for smooth appearance
- **Dual colormaps**: RdYlBu_r for error, plasma for identifiability

## Expected runtime

~30-60 seconds on modern hardware (toy backend, 112 runs).

## Use cases

1. **README screenshot**: Default hero image
2. **Presentations**: Dark mode figure
3. **Validation**: Verify QVL installation works
4. **Baseline**: Compare custom backends against canonical toy results

## Extending the battery

To modify the battery:

1. Edit `batteries/canonical_battery.yaml`
2. Adjust noise ranges, sample counts, or backend
3. Run `python -m qvl battery`

To create a custom battery:

1. Copy `batteries/canonical_battery.yaml` to `batteries/my_battery.yaml`
2. Copy `scripts/run_canonical_battery.py` to `scripts/run_my_battery.py`
3. Update paths and parameters
4. Run your custom battery

## CSV format

The summary CSV contains:

| Column | Description |
|--------|-------------|
| `depolarizing_p` | Feature noise level |
| `measurement_bitflip_p` | Label noise level |
| `accuracy` | Classification accuracy [0, 1] |
| `error` | 1 - accuracy |
| `loss` | Training loss |
| `ident_proxy` | Identifiability proxy [0, 1] |
| `fisher_condition_number` | Fisher matrix conditioning |
| `fisher_effective_rank` | Effective parameter rank |
| `hessian_min_abs` | Min Hessian eigenvalue |
| `hessian_max_abs` | Max Hessian eigenvalue |
| `wall_time_sec` | Execution time |

## Theory

The verification gap occurs because:

1. **Noise adds randomness** that models can exploit without learning true patterns
2. **Parameter space collapses** as Fisher Information Matrix becomes ill-conditioned
3. **Identifiability fails** when many parameter settings produce similar outputs
4. **Accuracy masks the problem** because test performance appears acceptable

The battery makes this invisible failure mode visible and quantifiable.
