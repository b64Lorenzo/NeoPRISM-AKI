
"""CLI registration for parameter extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.parameters.extractor import (
    extract_parameters,
)
from src.utils.logger import log


def register_parser(
    commands: argparse._SubParsersAction,
) -> None:
    parser = commands.add_parser(
        "parameters-extraction",
        help=(
            "Extract configured parameters into "
            "per-patient, phase-separated CSV files."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Input CSV files or directories.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/parameters.ini"),
        help="INI parameter configuration.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/parameters"),
        help="Output directory.",
    )

    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    log(
        "INFO",
        "parameters",
        "Starting parameter extraction",
    )

    stats = extract_parameters(
        inputs=args.input,
        output_dir=args.output_dir,
        config_path=args.config,
    )

    log(
        "INFO",
        "parameters",
        (
            f"Completed: files={stats['files']}; "
            f"patients={stats['patients']}; "
            f"parameters={stats['parameters']}; "
            "parameter-phase assignments="
            f"{stats['parameter_phase_assignments']}; "
            f"readings={stats['readings']}; "
            f"derived={stats['derived_readings']}"
        ),
    )

    return 0
