"""Command-line interface."""

import argparse
import sys
from pathlib import Path

from .config import load_config
from .runner import run_single_experiment, run_sweep


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Quantum Machine Learning Verification Laboratory',
        prog='qvl'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    run_parser = subparsers.add_parser('run', help='Run a single experiment')
    run_parser.add_argument('--config', required=True, help='Path to config file')
    run_parser.add_argument('--output-dir', default='artifacts', help='Output directory')
    run_parser.add_argument('--seed', type=int, default=0, help='Random seed')
    run_parser.add_argument('--quiet', action='store_true', help='Suppress output')

    sweep_parser = subparsers.add_parser('sweep', help='Run a parameter sweep')
    sweep_parser.add_argument('--config', required=True, help='Path to config file')
    sweep_parser.add_argument('--output-dir', default='artifacts', help='Output directory')
    sweep_parser.add_argument('--seeds', default='0', help='Comma-separated list of seeds')
    sweep_parser.add_argument('--quiet', action='store_true', help='Suppress output')

    report_parser = subparsers.add_parser('report', help='Generate report (stub)')
    report_parser.add_argument('--input-dir', help='Input directory with artifacts')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'run':
        return run_command(args)
    elif args.command == 'sweep':
        return sweep_command(args)
    elif args.command == 'report':
        return report_command(args)
    else:
        parser.print_help()
        return 1


def run_command(args):
    """Handle 'run' command."""
    try:
        config = load_config(args.config)
        run_single_experiment(
            config,
            args.output_dir,
            args.seed,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def sweep_command(args):
    """Handle 'sweep' command."""
    try:
        config = load_config(args.config)

        seeds = [int(s.strip()) for s in args.seeds.split(',')]

        run_sweep(
            config,
            args.output_dir,
            seeds,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def report_command(args):
    """Handle 'report' command (stub for Prompt D)."""
    print("Report generation is not yet implemented.")
    print("This feature will be added in a future update.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
