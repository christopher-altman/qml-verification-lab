# Quantum Machine Learning Verification Laboratory (QVL)

*A reproducible verification harness for Quantum Machine Learning systems that challenges the fundamental assumption that **accuracy is evidence of learning**.*

<br>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Profile-blue?logo=google-scholar)](https://scholar.google.com/citations?user=tvwpCcgAAAAJ)
[![Hugging Face](https://img.shields.io/badge/huggingface-Cohaerence-white)](https://huggingface.co/Cohaerence)
[![X](https://img.shields.io/badge/X-@coherence-blue)](https://x.com/coherence)
[![Website](https://img.shields.io/badge/website-christopheraltman.com-green)](https://www.christopheraltman.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Altman-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/Altman)

<br>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/hero_dark.jpg">
    <source media="(prefers-color-scheme: light)" srcset="figures/hero_light.jpg">
    <img alt="The Verification Gap: High accuracy does not imply parameter identifiability." src="figures/hero_light.jpg" width="100%">
  </picture>
</div>

<br>

---

## Table of Contents

1. [The Phenomenon](#the-phenomenon)
2. [Theoretical Basis](#theoretical-basis)
3. [Installation](#installation)
4. [Quick Start: Batteries & Sweeps](#quick-start-batteries--sweeps)
5. [Architecture](#architecture)
6. [Artifact Contract](#artifact-contract)
7. [Reproduction](#reproduction)
8. [References](#references)
9. [Citations](#citations)

---

## The Phenomenon

In noisy quantum systems, traditional validation metrics fail. We observe a troubling pattern where models maintain high test accuracy while their parameter-to-output mapping becomes many-to-one. This is the **Verification Gap**.

A model inside this gap is scientifically invalid:

1. **Identifiability Collapse:** Parameters are not uniquely determined by data.  
2. **Geometry Degeneration:** The Fisher Information Matrix becomes singular.  
3. **False Confidence:** The model exploits specific noise signatures rather than learning generalizable features.

QVL provides the diagnostics to quantify this gap.

## Theoretical Basis

QVL moves beyond loss curves to measure the **Epistemic Quality** of the learned model using *information geometry* (the study of how probability models form curved “surfaces” in parameter space).

### Fisher Information & Identifiability

We utilize the Fisher Information Matrix (FIM), $I(\theta)$, which quantifies the amount of information the observable data $X$ carries about parameters $\theta$:

$$
I(\theta)_{ij} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{\partial \log p(x\mid\theta)}{\partial \theta_i} \frac{\partial \log p(x\mid\theta)}{\partial \theta_j} \right]
$$

When $I(\theta)$ is rank-deficient or ill-conditioned (high condition number), the model is **non-identifiable**—changes in parameters along certain directions do not affect the output.

### Effective Dimension

To measure the “active” capacity of the model under noise, we compute an Effective Dimension based on the trace of the normalized FIM eigenvalues $\lambda_k$:

$$
d_{\mathrm{eff}} = \frac{N \sum_k \lambda_k}{\sum_k \lambda_k + \delta}
$$

Where $N$ is the number of data points and $\delta$ is a regularization term. A collapse in $d_{\mathrm{eff}}$ indicates that noise has washed out the model’s expressivity, even if accuracy remains high.

## Installation

**Basic installation (toy & deterministic backends):**

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

## Quick Start: Batteries & Sweeps

### 1) The Canonical Battery

The fastest way to demonstrate the verification gap. Runs a standardized 16×7 noise sweep (112 runs) to generate the “Hero Plot” showing the divergence of accuracy and identifiability.

```bash
python -m qvl battery
```

Outputs: `results/summary.csv` and `figures/hero_*.png`.

### 2) Custom Parameter Sweeps

Run targeted experiments defined in YAML configuration.

**Toy backend (classical logistic regression):**

```bash
python -m qvl sweep --config examples/toy_sweep_small.yaml --output-dir artifacts/ --seeds 0,1,2
```

**PennyLane backend (VQC):**

```bash
python -m qvl sweep --config examples/vqc_sweep_small.yaml --output-dir artifacts/ --seeds 0,1
```

### 3) Reporting

Generate comprehensive reports with leaderboards, heatmaps, and storytelling figures.

## Architecture

QVL is designed as a rigorous scientific instrument with a clean separation of concerns.

```text
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

### Backend Plugin System

Backends implement a strict interface: `__init__(config)` and `run() -> dict`.

- `deterministic`: NumPy-only, bit-perfect reproduction for CI/testing
- `toy`: Synthetic logistic regression with controllable noise channels
- `pennylane`: Variational quantum classifiers with depolarization and readout error

### Verification Metrics

| Metric                    | Purpose                        | Warning threshold            |
|--------------------------|--------------------------------|------------------------------|
| `ident_proxy`            | Identifiability signal strength | `< 0.1` (with `acc > 0.7`)   |
| `fisher_condition_number`| Parameter determinability       | `> 1000`                     |
| `fisher_effective_rank`  | Active parameter dimensions     | `< 0.5 × theoretical`        |
| `hessian_trace`          | Loss landscape curvature        | N/A                          |
| `seed_robustness`        | Stability across initialization | Variance `> 0.1`             |

## Artifact Contract

QVL enforces a **stable artifact contract**. Every run, regardless of backend, produces a deterministic structure. This guarantees that downstream analysis tools never break.

```text
artifacts/<experiment_id>/run_seed{seed:04d}_{hash}/
├── config.resolved.json          # Full frozen configuration
├── summary.json                  # Standardized metrics (Schema: v1)
├── results.jsonl                 # Per-shot / per-epoch data
├── env.json                      # Environment snapshot
└── figures/
    ├── hero_identifiability_dark.png
    ├── hero_identifiability_light.png
```

**Warning trigger:** The system automatically flags runs where `accuracy > 0.7` but `ident_proxy < 0.1` as “Potential Overfitting / Noise Dominance”.

## Reproduction

Reproducibility is treated as a first-class constraint.

1. **Seed control:** All experiments accept `--seed`.  
2. **Config hashing:** Output directories include a hash of the resolved config (e.g., `run_seed0042_2ad599bf`).  
3. **Artifact separation:**
   - `artifacts_demo/`: tracked in git (examples and baselines)
   - `artifacts/`: gitignored (local experimentation)
   - **CI artifacts:** generated on every push, retained for 7 days

To regenerate the demo artifacts found in this repo:

```bash
scripts/generate_demo_artifacts.sh
```

## Tags

`qml` · `verification` · `identifiability` · `fisher-information` · `robustness` · `quantum-computing` · `machine-learning` · `reproducibility` · `harness` · `noise-analysis`

## References

1. Abbas, A., et al. (2021). The power of quantum neural networks. *Nature Computational Science*, 1(6), 403–409.
2. Cerezo, M., et al. (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625–644.
3. Holmes, Z., et al. (2022). Connecting ansatz expressibility to gradient magnitudes and barren plateaus. *PRX Quantum*, 3(1), 010313.
4. LaRose, R., & Coyle, B. (2020). Robust data encodings for quantum classifiers. *Physical Review A*, 102(3), 032420.
5. Sharma, K., et al. (2022). Reformulation of the no-free-lunch theorem for entangled datasets. *Physical Review Letters*, 128(7), 070501.

## Citations

If you use QVL in your research, please cite:

```bibtex
@software{qvl2026,
  title        = {Quantum Machine Learning Verification Laboratory},
  author       = {Altman, Christopher},
  year         = {2026},
  url          = {https://github.com/christopher-altman/qml-verification-lab}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Website:** [christopheraltman.com](https://christopheraltman.com)
- **Research portfolio:** https://lab.christopheraltman.com/
- **Portfolio mirror:** https://christopher-altman.github.io/
- **GitHub:** https://github.com/christopher-altman
- **Google Scholar:** https://scholar.google.com/citations?user=tvwpCcgAAAAJ
- **Email:** x@christopheraltman.com

---

*Christopher Altman (2026)*