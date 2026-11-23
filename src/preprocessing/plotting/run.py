
"""Standalone plotting command."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.parameters.config import (
    load_config as load_parameter_config,
)
from src.preprocessing.plotting.config import (
    load_config as load_plot_config,
)
from src.preprocessing.plotting.data import (
    load_observations, load_patient_boundaries, load_patient_events,
)
from src.preprocessing.plotting.patient_dashboard import generate_patient_dashboards
from src.preprocessing.plotting.plots import (
    plot_distributions,
    plot_frequencies,
    plot_missingness,
    plot_patient_timelines,
    plot_time_series,
    plot_value_heatmap,
)
from src.preprocessing.plotting.selection import (
    build_catalog,
    select_parameters,
    select_heatmap_parameters,
)
from src.utils.logger import log


def register_parser(
    commands: argparse._SubParsersAction,
) -> None:
    parser = commands.add_parser(
        "plotting",
        help="Generate plots from extracted patient parameters.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/plotting.ini"),
        help="Plotting INI configuration.",
    )

    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    config = load_plot_config(args.config)

    parameter_config = load_parameter_config(
        config.parameter_config
    )

    catalog = build_catalog(
        parameter_config.parameters,
        config,
    )
    selected = select_parameters(catalog, config)

    heatmap_catalog = (
        select_heatmap_parameters(
            catalog, config.heatmap_phases, config.heatmap_parameter_ids,
            numeric_only=True, label="value_heatmaps",
        )
        if config.enabled["value_heatmaps"] else {}
    )
    missingness_catalog = (
        select_heatmap_parameters(
            catalog, config.missingness_phases, config.missingness_parameter_ids,
            label="missingness",
        )
        if config.enabled["missingness"] else {}
    )
    availability_catalog = (
        select_heatmap_parameters(
            catalog, config.availability_phases, config.availability_parameter_ids,
            label="parameter_availability",
        ) if config.enabled["parameter_availability"] else {}
    )
    load_catalog = dict(selected)
    load_catalog.update(heatmap_catalog)
    load_catalog.update(missingness_catalog)
    load_catalog.update(availability_catalog)
    observations, patients = load_observations(
        config.input_dir,
        load_catalog,
        config.selection.patient_ids,
    )

    ordinary_observations = [row for row in observations if row.key in selected]
    missingness_observations = [row for row in observations if row.key in missingness_catalog]
    availability_observations = [row for row in observations if row.key in availability_catalog]
    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if config.patient_dashboards_enabled:
        count = generate_patient_dashboards(
            ordinary_observations, selected, patients, config.patient_dashboards_dir,
            theme=config.theme, include_plotlyjs=config.patient_dashboards_embed_js,
            boundaries=load_patient_boundaries(config.input_dir, patients),
            events=load_patient_events(config.input_dir, patients),
        )
        log("INFO", "plotting", f"patient_dashboards: {count} HTML files generated")

    operations = {
        "distributions": lambda: plot_distributions(
            ordinary_observations, selected, config
        ),
        "frequencies": lambda: plot_frequencies(
            ordinary_observations, selected, config
        ),
        "missingness": lambda: plot_missingness(
            missingness_observations, missingness_catalog, patients, config
        ),
        "parameter_availability": lambda: plot_missingness(
            availability_observations, availability_catalog, patients, config,
            filename="availability.html", title="Parameter availability by patient and phase",
        ),
        "value_heatmaps": lambda: plot_value_heatmap(
            observations, heatmap_catalog, patients, config
        ),
        "time_series": lambda: plot_time_series(
            ordinary_observations, selected, config
        ),
        "patient_timelines": lambda: plot_patient_timelines(
            ordinary_observations, selected, patients, config
        ),
    }

    total = 0

    for name, operation in operations.items():
        if not config.enabled[name]:
            continue

        log("INFO", "plotting", f"Generating {name}")

        count = operation()
        total += count

        log(
            "INFO",
            "plotting",
            f"{name}: {count} plots generated",
        )

    log(
        "INFO",
        "plotting",
        (
            f"Completed; plots={total}; "
            f"patients={len(patients)}; "
            f"parameters={len(selected)}; "
            f"observations={len(observations)}"
        ),
    )

    return 0
