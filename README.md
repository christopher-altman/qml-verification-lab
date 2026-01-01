# Quantum Machine Learning Verification Laboratory (QVL)

*A reproducible verification harness for Quantum Machine Learning systems that challenges a fundamental assumption:* ***accuracy is not evidence of learning***.

<br>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Profile-blue?logo=google-scholar)](https://scholar.google.com/citations?user=tvwpCcgAAAAJ)
[![Hugging Face](https://img.shields.io/badge/huggingface-Cohaerence-white)](https://huggingface.co/Cohaerence)

[![X](https://img.shields.io/badge/X-@coherence-blue)](https://x.com/coherence)
[![Website](https://img.shields.io/badge/website-christopheraltman.com-green)](https://www.christopheraltman.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Altman-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/Altman)

## The Problem

High test accuracy in quantum machine learning models can arise from multiple mechanisms:

1. **True learning**: The model has discovered generalizable patterns
2. **Noise exploitation**: The model exploits specific noise signatures rather than signal
3. **Identifiability collapse**: Parameters are not uniquely determined by data

Standard ML evaluation cannot distinguish between these cases. A model can achieve high accuracy while being fundamentally non-identifiable, making it scientifically invalid and practically unreliable.

## The Phenomenon

In noisy quantum systems, we observe a troubling pattern:

- Models maintain high accuracy under noise
- But the parameter-to-output mapping becomes many-to-one
- Fisher information geometry collapses
- Robustness to seed/initialization vanishes
- The model appears to "work" but is not actually learning

This is not a bug. It's a fundamental challenge for QML verification.

## Our Hypothesis

We hypothesize that QML verification requires a battery of diagnostics beyond accuracy:

**Identifiability**: Can we uniquely recover parameters from observations?
- Fisher information matrix properties
- Parameter space geometry
- Effective dimensionality

**Information Geometry**: What does the loss landscape reveal?
- Hessian eigenspectrum
- Condition numbers
- Local curvature

**Robustness**: Is the solution stable?
- Seed sensitivity
- Noise perturbation response
- Initialization dependence

## Method

QVL implements a systematic verification protocol:

1. **Controlled noise injection** at multiple levels (depolarizing, measurement, amplitude)
2. **Battery metrics** computed on every run (identifiability proxies, Fisher diagnostics, Hessian analysis)
3. **Sweep automation** over noise × seed grids
4. **Stable artifact contracts** for reproducibility
5. **Verification warnings** when accuracy and identifiability diverge

### Noise Parameters

- `depolarizing_p`: Feature noise intensity (simulates decoherence)
- `measurement_bitflip_p`: Label noise (simulates measurement errors)
- `amplitude_gamma`: Regularization strength (simulates amplitude damping)

### Verification Metrics

| Metric | Purpose | Warning Threshold |
|--------|---------|-------------------|
| `accuracy` | Standard performance | N/A |
| `ident_proxy` | Identifiability signal | < 0.1 with accuracy > 0.7 |
| `fisher_condition_number` | Parameter determinability | > 1000 |
| `fisher_effective_rank` | Active parameter dimensions | < 0.5 × theoretical |
| `hessian_min_abs` / `hessian_max_abs` | Loss landscape geometry | Ratio > 1000 |
| `seed_robustness` | Stability across seeds | Variance > 0.1 |

## Implementation

### Architecture

```
qvl/
├── cli.py              # Argument parsing and command dispatch
├── config.py           # YAML loading and validation
├── runner.py           # Single run + sweep orchestration
├── artifacts.py        # Standardized output contract
├── plotting.py         # Hero figures (dark/light variants)
├── batteries/          # Verification metric implementations
└── backends/           # Pluggable experiment backends
    └── toy/            # Toy logistic regression (Prompt A)
```

### Artifact Contract

Every run produces a stable directory structure:

```
artifacts/<experiment_id>/run_<timestamp>_<hash>/
├── config.resolved.json          # Full resolved configuration
├── summary.json                  # Standardized metrics (stable schema)
├── results.jsonl                 # Per-point results (for sweeps)
├── env.json                      # Environment snapshot
├── git.json                      # Git metadata (optional)
├── tables/
│   └── leaderboard.csv
└── figures/
    ├── hero_identifiability_dark.png
    ├── hero_identifiability_light.png
    ├── hero_identifiability_dark_transparent.png
    └── hero_identifiability_light_transparent.png
```

The `summary.json` schema is **guaranteed stable** for downstream automation.

### Backend Plugin System

Backends implement a simple interface:

```python
class Backend:
    def __init__(self, config: dict): ...
    def run(self) -> dict: ...  # Returns metrics, noise, timing
```

**Available backends:**
- **`toy`**: Synthetic logistic regression with controlled noise (always available)
- **`pennylane`**: 2-qubit VQC with quantum noise channels (optional, requires `pip install -e ".[pennylane]"`)
- **`template`**: Non-functional stub demonstrating the backend interface

**Add your own backend:** See [BACKENDS.md](BACKENDS.md) for the complete guide (< 60 seconds to get started).

**Future backends:**
- `qiskit`: IBM quantum simulators
- `cirq`: Google quantum frameworks

## Usage

### Installation

**Basic installation (toy backend only):**
```bash
git clone https://github.com/christopher-altman/qml-verification-lab.git
cd qml-verification-lab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**With PennyLane backend:**
```bash
pip install -e ".[dev,pennylane]"
```

### Quick Start

**Toy backend (classical):**
```bash
# Single run
python -m qvl run --config examples/toy_smoke.yaml --output-dir artifacts/

# Parameter sweep
python -m qvl sweep --config examples/toy_sweep_small.yaml --output-dir artifacts/ --seeds 0,1,2
```

**PennyLane backend (quantum):**
```bash
# Single run
python -m qvl run --config examples/vqc_smoke.yaml --output-dir artifacts/

# Parameter sweep
python -m qvl sweep --config examples/vqc_sweep_small.yaml --output-dir artifacts/ --seeds 0,1
```

**Help:**
```bash
python -m qvl --help
```

### Reporting

After running sweeps, generate comprehensive reports with leaderboards and visualizations:

```bash
python -m qvl report --input artifacts/my_experiment_id/ --output reports/my_report/
```

**Report Outputs:**
- `leaderboard.csv` - Tabular results with stable columns (experiment_id, backend, accuracy, ident_proxy, fisher metrics, noise parameters, seed, timing, run_dir)
- `figures/accuracy_vs_identifiability.png` - Scatter plot showing accuracy vs identifiability colored by Fisher condition number
- `figures/fisher_vs_accuracy.png` - Fisher condition number vs accuracy plot
- `figures/identifiability_heatmap.png` - Heatmap of identifiability proxy across noise grid (if grid-like sweep detected)
- `hero_dark.png` / `hero_light.png` - Hero images from automatically selected "storytelling point" (high accuracy but low identifiability tension)

**Hero Selection Heuristic:**
The report automatically selects a hero point for storytelling by scoring each result:
- Rewards high accuracy
- Penalizes low identifiability and high Fisher condition number
- Bonus for "tension points": high accuracy (>0.7) but low identifiability (<0.3)

This identifies the most compelling demonstration of the verification gap.

### Example Configuration

```yaml
experiment_id: my_verification_run
backend: toy
task: classification

training:
  n_samples: 200
  n_features: 2
  n_epochs: 150
  learning_rate: 0.1

noise:
  depolarizing_p: 0.1
  measurement_bitflip_p: 0.05
  amplitude_gamma: 0.01

# For sweeps
sweep:
  depolarizing_p: [0.0, 0.05, 0.1, 0.15, 0.2]
  measurement_bitflip_p: [0.0, 0.05, 0.1, 0.15, 0.2]
```

## Initial Results (Toy Backend)

Even in the toy backend, we observe the core phenomenon:

- At `depolarizing_p = 0.0, measurement_bitflip_p = 0.0`: accuracy ≈ 0.95, identifiability ≈ 0.8
- At `depolarizing_p = 0.2, measurement_bitflip_p = 0.2`: accuracy ≈ 0.70, identifiability ≈ 0.05

**Verification warning triggered**: "High accuracy but low identifiability - potential overfitting or noise dominance"

This demonstrates that noise can degrade identifiability faster than accuracy, invalidating the learned model despite acceptable test performance.

## Interpretation

### Why Accuracy Alone is Insufficient

Traditional ML assumes:
- Unique parameter-to-output mapping (identifiability)
- Smooth, well-conditioned loss landscape
- Stability across random initialization

Quantum systems violate all three under noise. A high-accuracy QML model might be:
- Non-identifiable (many parameter settings produce same output)
- Ill-conditioned (Fisher information matrix near-singular)
- Non-robust (different seeds give different solutions)

These failures are invisible to accuracy metrics but fatal to scientific validity.

### The Verification Gap

The gap between accuracy and identifiability reveals the **verification gap**: the space where a model appears to work but is not actually learning. QVL makes this gap quantifiable and reproducible.

## Artifacts Policy

QVL follows a clean separation between source code and generated experimental outputs:

**Tracked (in git):**
- `artifacts_demo/` - Small curated snapshot (~200 KB) for quick review
  - Contains 1-2 example runs with hero plots and summary metrics
  - See `artifacts_demo/README.md` for details

**Not tracked (generated locally):**
- `artifacts/` - Full experiment outputs (gitignored)
- Generated via `qvl run` or `qvl sweep` commands

**Regenerating demo artifacts:**
```bash
bash scripts/generate_demo_artifacts.sh
```

**CI artifacts:**
- GitHub Actions uploads full test artifacts (not committed to repo)
- Available as downloadable CI artifacts on each workflow run

This policy keeps the repository lightweight while maintaining reproducibility.

## Development Roadmap

- **Phase 1.0** ✓ MVP with toy backend, stable artifacts, hero plots
- **Phase 2.0** ✓ Real Fisher information, Hessian computation (logistic regression)
- **Phase 3.0** ✓ PennyLane backend with quantum noise channels (current)
- **Phase 4.0**: Extended robustness batteries, cross-seed aggregation
- **Phase 5.0**: Interactive reports, web dashboard, publication-ready figures

## Lineage

This laboratory extends and generalizes prior experimental work:

**[noise-aware-qnn-identifiability](https://github.com/christopher-altman/noise-aware-qnn-identifiability)** (2024)
- Demonstrated that quantum neural networks under noise can exhibit high accuracy while losing parameter identifiability
- Used Fisher Information Matrix rank deficiency as the primary diagnostic
- Implemented proof-of-concept with PennyLane on fixed circuit architectures
- Established the verification gap: accuracy does not imply learning in noisy quantum systems

QVL generalizes this finding by:
- Abstracting the backend (toy, PennyLane, future: Qiskit, Cirq)
- Formalizing identifiability and curvature as first-class metrics via the metric registry
- Adding systematic noise sweeps, reproducible artifact contracts, and automated reporting
- Expanding verification diagnostics beyond Fisher rank to include Hessian geometry, robustness batteries, and warning thresholds

The core hypothesis remains unchanged: **accuracy is not evidence of learning**. QVL provides the infrastructure to test this hypothesis systematically across backends and noise regimes.

## Tags

`qml` · `verification` · `identifiability` · `fisher-information` · `robustness` · `quantum-computing` · `machine-learning` · `reproducibility` · `harness` · `noise-analysis`

## References

1. Abbas, A., et al. (2021). The power of quantum neural networks. *Nature Computational Science*, 1(6), 403-409.
2. Cerezo, M., et al. (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625-644.
3. Holmes, Z., et al. (2022). Connecting ansatz expressibility to gradient magnitudes and barren plateaus. *PRX Quantum*, 3(1), 010313.
4. LaRose, R., & Coyle, B. (2020). Robust data encodings for quantum classifiers. *Physical Review A*, 102(3), 032420.
5. Sharma, K., et al. (2022). Reformulation of the no-free-lunch theorem for entangled datasets. *Physical Review Letters*, 128(7), 070501.

## Citations

If you use QVL in your research, please cite:

```bibtex
@software{qvl2026,
  title={Quantum Machine Learning Verification Laboratory},
  author={Altman, Christopher},
  year={2026},
  url={https://github.com/christopher-altman/qml-verification-lab}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Website:** [christopheraltman.com](https://christopheraltman.com)
- **Research portfolio:** https://lab.christopheraltman.com/
- **Portfolio mirror:** https://christopher-altman.github.io/
- **GitHub:** [github.com/christopher-altman](https://github.com/christopher-altman)
- **Google Scholar:** [scholar.google.com/citations?user=tvwpCcgAAAAJ](https://scholar.google.com/citations?user=tvwpCcgAAAAJ)
- **Email:** x@christopheraltman.com

---

*Christopher Altman (2025)*
