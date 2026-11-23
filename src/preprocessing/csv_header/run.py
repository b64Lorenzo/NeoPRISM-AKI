"""CLI entry point for CSV header inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.csv_header.inventory import write_csv_header_descriptions
from src.utils.logger import log
from src.preprocessing.csv_header.events import write_event_reports


def register_parser(
    commands: argparse._SubParsersAction,
) -> None:
    parser = commands.add_parser(
        "csv-header",
        help="Inspect CSV headers without reading patient records.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Input CSV files or directories.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/csv_header"),
        help="Directory for generated reports.",
    )

    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=None,
        help="Optional path for the inventory CSV.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path for the Markdown summary.",
    )

    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()

    inventory = (
        args.inventory_output
        if args.inventory_output is not None
        else output_dir / "header_inventory.csv"
    )

    summary = (
        args.summary_output
        if args.summary_output is not None
        else output_dir / "header_summary.md"
    )

    log("INFO", "csv_header", "Starting CSV header analysis")

    stats = write_csv_header_descriptions(
        inputs=args.input,
        output_directory=output_dir,
        inventory_output=inventory,
        summary_output=summary,
    )

    log(
        "INFO",
        "csv_header",
        f"Processed {stats['files']} CSV files; "
        f"found {stats['unique_headers']} distinct headers",
    )

    log("INFO", "csv_header", f"Inventory: {inventory.resolve()}")
    log("INFO", "csv_header", f"Summary: {summary.resolve()}")

    event_stats = write_event_reports(args.input, output_dir)
    log("INFO", "csv_header", f"Event strings: {event_stats['event_values']}; observations: {event_stats['event_observations']}")

    return 0
