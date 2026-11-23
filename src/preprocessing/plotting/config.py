
"""INI configuration for the standalone plotting command."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


PHASES = {"preoperative", "intraoperative", "postoperative"}
PLOT_TYPES = (
    "distributions",
    "frequencies",
    "missingness",
    "parameter_availability",
    "value_heatmaps",
    "time_series",
    "patient_timelines",
)


def items(value: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class Selection:
    phases: tuple[str, ...]
    parameter_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    families: tuple[str, ...]
    patient_ids: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class PlotConfig:
    input_dir: Path
    parameter_config: Path
    output_dir: Path
    theme: str
    max_patients: int
    max_categories: int
    selection: Selection
    families: dict[str, tuple[str, ...]]
    enabled: dict[str, bool]
    bins: int
    numeric_aggregation: str
    normalize_frequencies: bool
    heatmap_aggregation: str
    heatmap_parameter_ids: tuple[str, ...]
    missingness_parameter_ids: tuple[str, ...]
    heatmap_phases: tuple[str, ...]
    missingness_phases: tuple[str, ...]
    availability_parameter_ids: tuple[str, ...]
    availability_phases: tuple[str, ...]
    time_series_group_by: str
    time_series_max_patients: int
    timeline_max_patients: int
    patient_dashboards_enabled: bool
    patient_dashboards_dir: Path
    patient_dashboards_embed_js: bool


def load_config(path: str | Path) -> PlotConfig:
    path = Path(path).resolve()
    parser = configparser.ConfigParser(interpolation=None)

    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(path)

    def get(section: str, option: str, fallback="") -> str:
        return parser.get(section, option, fallback=fallback)

    def resolved(value: str) -> Path:
        candidate = Path(value)
        # Paths are relative to the project root, assuming
        # config/plotting.ini is inside the config directory.
        root = path.parent.parent
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )

    phases = items(
        get("selection", "phases", ",".join(sorted(PHASES)))
    )
    if set(phases) - PHASES:
        raise ValueError("Invalid plotting phase")

    def plot_phases(section: str) -> tuple[str, ...]:
        requested = tuple(value.casefold() for value in items(get(section, "phases", "all")))
        if not requested or requested == ("all",):
            return ("preoperative", "intraoperative", "postoperative")
        if "all" in requested or len(requested) != len(set(requested)) or set(requested) - PHASES:
            raise ValueError(f"[{section}] phases must be 'all' or distinct valid phases")
        return requested

    mode = get("selection", "mode", "union").lower()
    if mode not in {"union", "intersection"}:
        raise ValueError("Selection mode must be union or intersection")

    theme = get("general", "theme", "light").lower()
    if theme not in {"light", "dark"}:
        raise ValueError("Plot theme must be light or dark")

    if get("general", "format", "html").lower() != "html":
        raise ValueError("Only HTML output is currently supported")

    aggregation = get(
        "distributions", "numeric_aggregation", "patient_mean"
    )
    heatmap_aggregation = get(
        "value_heatmaps", "aggregation", "patient_mean"
    )
    valid_aggregations = {
        "patient_mean", "patient_median",
        "patient_min", "patient_max", "all_observations",
    }
    if aggregation not in valid_aggregations:
        raise ValueError("Invalid distribution aggregation")
    if heatmap_aggregation not in valid_aggregations - {"all_observations"}:
        raise ValueError("Invalid heatmap aggregation")

    group_by = get("time_series", "group_by", "parameter")
    if group_by not in {"parameter", "class", "family"}:
        raise ValueError("Invalid time-series group_by")

    families = {
        name: items(value)
        for name, value in parser.items("families")
    } if parser.has_section("families") else {}

    enabled = {
        name: parser.getboolean(
            name, "enabled", fallback=False
        )
        for name in PLOT_TYPES
    }

    result = PlotConfig(
        input_dir=resolved(get("paths", "input_directory")),
        parameter_config=resolved(
            get("paths", "parameter_config")
        ),
        output_dir=resolved(get("paths", "output_directory")),
        theme=theme,
        max_patients=parser.getint(
            "general", "max_patients", fallback=500
        ),
        max_categories=parser.getint(
            "general", "max_categories", fallback=30
        ),
        selection=Selection(
            phases=phases,
            parameter_ids=items(
                get("selection", "parameter_ids")
            ),
            class_names=items(
                get("selection", "class_names")
            ),
            families=items(get("selection", "families")),
            patient_ids=items(
                get("selection", "patient_ids")
            ),
            mode=mode,
        ),
        families=families,
        enabled=enabled,
        bins=parser.getint(
            "distributions", "bins", fallback=30
        ),
        numeric_aggregation=aggregation,
        normalize_frequencies=parser.getboolean(
            "frequencies", "normalize", fallback=True
        ),
        heatmap_aggregation=heatmap_aggregation,
        heatmap_parameter_ids=items(get("value_heatmaps", "parameter_ids", "all")),
        missingness_parameter_ids=items(get("missingness", "parameter_ids", "all")),
        heatmap_phases=plot_phases("value_heatmaps"),
        missingness_phases=plot_phases("missingness"),
        availability_parameter_ids=items(get("parameter_availability", "parameter_ids", "all")),
        availability_phases=plot_phases("parameter_availability"),
        time_series_group_by=group_by,
        time_series_max_patients=parser.getint(
            "time_series", "max_patients", fallback=20
        ),
        timeline_max_patients=parser.getint(
            "patient_timelines", "max_patients", fallback=20
        ),
        patient_dashboards_enabled=parser.getboolean("patient_dashboards", "enabled", fallback=False),
        patient_dashboards_dir=resolved(get("patient_dashboards", "output_directory", "results/plots/patients")),
        patient_dashboards_embed_js=parser.getboolean("patient_dashboards", "include_plotlyjs", fallback=True),
    )

    if any(value < 1 for value in (
        result.max_patients,
        result.max_categories,
        result.bins,
        result.time_series_max_patients,
        result.timeline_max_patients,
    )):
        raise ValueError("Plot limits and bins must be positive")

    if result.input_dir == result.output_dir:
        raise ValueError("Plot input and output directories must differ")

    return result
