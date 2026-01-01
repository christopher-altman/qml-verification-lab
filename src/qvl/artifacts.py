"""Artifact management and writers."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class ArtifactWriter:
    """Manages artifact directory structure and writes."""

    def __init__(self, output_dir: str, experiment_id: str, config_hash: str, seed: int = 0):
        """Initialize artifact writer.

        Args:
            output_dir: Base output directory
            experiment_id: Experiment identifier
            config_hash: Short hash of configuration
            seed: Random seed for deterministic run naming
        """
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.config_hash = config_hash
        self.seed = seed

        # Deterministic run naming: seed-based, not timestamp-based
        # This ensures reproducible directory structure
        run_dir = f"run_seed{seed:04d}_{config_hash}"

        self.run_dir = self.output_dir / experiment_id / run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.tables_dir = self.run_dir / "tables"
        self.figures_dir = self.run_dir / "figures"
        self.tables_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

    def write_config(self, config: Dict[str, Any]) -> None:
        """Write resolved configuration."""
        path = self.run_dir / "config.resolved.json"
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def write_summary(self, summary: Dict[str, Any]) -> None:
        """Write summary JSON."""
        path = self.run_dir / "summary.json"
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

    def append_result(self, result: Dict[str, Any]) -> None:
        """Append result to JSONL file."""
        path = self.run_dir / "results.jsonl"
        with open(path, 'a') as f:
            f.write(json.dumps(result) + '\n')

    def write_env(self, env_info: Dict[str, Any]) -> None:
        """Write environment information."""
        path = self.run_dir / "env.json"
        with open(path, 'w') as f:
            json.dump(env_info, f, indent=2)

    def write_git_info(self, git_info: Dict[str, Any]) -> None:
        """Write git information (optional)."""
        path = self.run_dir / "git.json"
        with open(path, 'w') as f:
            json.dump(git_info, f, indent=2)

    def get_figure_path(self, name: str) -> Path:
        """Get path for figure."""
        return self.figures_dir / name

    def get_table_path(self, name: str) -> Path:
        """Get path for table."""
        return self.tables_dir / name


def get_env_info() -> Dict[str, Any]:
    """Collect environment information."""
    import platform
    import sys

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def try_get_git_info() -> Dict[str, Any]:
    """Try to get git information (returns empty dict if not available)."""
    import subprocess

    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        dirty = subprocess.call(
            ['git', 'diff-index', '--quiet', 'HEAD'],
            stderr=subprocess.DEVNULL
        ) != 0

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
