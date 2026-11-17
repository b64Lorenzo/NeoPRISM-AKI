"""Command-line entry point for the local CSV analysis PoC."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.logging.logger import Logger, LogLevel
from src.pipeline.runner import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETER_MAP = ROOT / "config" / "pipeline.xml"
DEFAULT_PIPELINE_CONFIG = ROOT / "config" / "pipeline.xml"


def _values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pattern-recognition",
        description="Build one pre-operative feature vector per patient CSV and assess outcomes from post-operative data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)

    run = commands.add_parser("run", help="run configured preprocessing and model stages from XML")
    run.add_argument("--config", type=Path, default=DEFAULT_PIPELINE_CONFIG, help="XML run configuration")

    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            message = run_pipeline(args.config)
            return 0
    except (csv.Error, OSError, ValueError, RuntimeError) as error:
        detail = str(error) if isinstance(error, (ValueError, RuntimeError)) else "check inputs, mapping, and dependencies"
        error_message = f"Command failed ({type(error).__name__}): {detail}"
        error_log_dir = getattr(args, "log_dir", ROOT / "logs")
        try:
            with Logger(minimum_level=LogLevel.ERROR, log_root=error_log_dir) as logger:
                logger.write(LogLevel.ERROR, "cli", error_message)
        except OSError:
            pass
        print(error_message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
