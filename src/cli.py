"""Main command-line interface for the AKI/KDIGO pipeline."""

from __future__ import annotations

import argparse
import csv
import sys

from src.models.rosenblatt_perceptron.run import register_parser as register_rosenblatt
from src.models.run import register_parser as register_classifiers
from src.pipeline.run import register_parser as register_pipeline
from src.preprocessing.csv_header.run import register_parser as register_csv_header
from src.preprocessing.inputsModel.run import register_parser as register_inputs_model
from src.preprocessing.parameters.run import (
    register_parser as register_parameters,
)
from src.preprocessing.plotting.run import (
    register_parser as register_plotting,
)
from src.utils.logger import log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pattern-recognition",
        description="AKI/KDIGO prediction pipeline.",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    register_csv_header(commands)
    register_parameters(commands)
    register_plotting(commands)
    register_inputs_model(commands)
    register_rosenblatt(commands)
    register_classifiers(commands)
    register_pipeline(commands)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        log("INFO", "cli", f"Starting command: {args.command}")

        result = args.handler(args)

        log("INFO", "cli", f"Completed command: {args.command}")
        return 0 if result is None else result

    except (csv.Error, OSError, ValueError, RuntimeError) as error:
        log("ERROR", "cli", f"{type(error).__name__}: {error}")
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
