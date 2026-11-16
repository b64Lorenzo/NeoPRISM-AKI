"""Generate per-parameter distribution plots and aggregate summaries."""

from __future__ import annotations

import csv
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from src.plotlib.plotly_output import plotly_template, write_plotly_html
from src.preprocessing.parameters.parameter_map import load_parameter_map
from src.preprocessing.parameters.settings import select_parameter_map
from src.preprocessing.plotting.selection import PlotSelection
from src.utils.csv_utils import number_component


def _number(value: object) -> float | None:
    return number_component(value)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _plot_distribution(
    path: Path,
    title: str,
    values: list[str],
    bins: int,
    *,
    numeric_expected: bool | None = None,
    theme: str = "light",
) -> tuple[str, list[float], Counter[str]]:
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError("Distribution plots require Plotly; install requirements.txt") from error

    observed = [value for value in values if value.strip()]
    numeric = [_number(value) for value in observed]
    is_numeric = (
        bool(observed) and all(value is not None for value in numeric)
        if numeric_expected is None
        else numeric_expected
    )
    figure = go.Figure()
    categories: Counter[str] = Counter()
    numeric_values = [value for value in numeric if value is not None] if is_numeric else []

    if not observed:
        kind = "no_data"
        message = "No mapped values available"
    elif is_numeric and not numeric_values:
        kind = "no_data"
        message = "No numeric values available"
    elif is_numeric:
        kind = "numeric"
        message = ""
        figure.add_trace(
            go.Histogram(
                x=numeric_values,
                nbinsx=bins,
                marker_color="#4c78a8",
                marker_line_color="white",
                hovertemplate="Value: %{x}<br>Patient count: %{y}<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Recorded value")
        figure.update_yaxes(title_text="Patient count")
    else:
        kind = "categorical"
        message = ""
        categories = Counter(observed)
        ordered = sorted(categories.items(), key=lambda item: (-item[1], item[0]))
        labels = [item[0] for item in ordered]
        figure.add_trace(
            go.Bar(
                x=labels,
                y=[item[1] for item in ordered],
                marker_color="#4c78a8",
                hovertemplate="Category: %{x}<br>Patient count: %{y}<extra></extra>",
            )
        )
        figure.update_xaxes(title_text="Recorded category", tickangle=-45, automargin=True)
        figure.update_yaxes(title_text="Patient count")

    if message:
        figure.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    figure.update_layout(
        title=title,
        template=plotly_template(theme),
        showlegend=False,
        height=450,
        bargap=0.08,
        margin={"l": 75, "r": 30, "t": 80, "b": 110},
    )
    write_plotly_html(figure, path)
    return kind, numeric_values, categories


def _measurement_column(parameter: Mapping, postoperative_specs: list[dict], available: set[str]) -> str:
    parameter_id = str(parameter["id"])
    if parameter_id in available:
        return parameter_id
    wanted_sources = set(parameter.get("source_fields", []))
    matches = [
        str(spec["id"])
        for spec in postoperative_specs
        if wanted_sources.intersection(spec.get("source_fields", [])) and spec.get("id") in available
    ]
    return matches[0] if len(matches) == 1 else ""


def _format(value: float) -> str:
    return format(value, ".8g")


def generate_parameter_distributions(
    output_dir: str | Path,
    parameter_map_path: str | Path | Mapping,
    *,
    bins: int = 20,
    phase: str = "all",
    parameter_ids: Iterable[str] = (),
    class_names: Iterable[str] = (),
    theme: str = "light",
    parameter_selection: Mapping[str, object] | None = None,
) -> int:
    """Plot every mapped parameter from the patient-level analysis tables."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    output = Path(output_dir)
    config = (
        parameter_map_path
        if isinstance(parameter_map_path, Mapping)
        else load_parameter_map(parameter_map_path)
    )
    if parameter_selection is not None:
        config = select_parameter_map(config, **parameter_selection)
    parameters = PlotSelection(phase.casefold(), list(parameter_ids), list(class_names)).select(config)
    postoperative_specs = config.get("postoperative_measurements", [])
    preoperative_rows = _read_rows(output / "patient_vectors.csv")
    duringoperative_rows = _read_rows(output / "duringoperative_measurements.csv")
    postoperative_rows = _read_rows(output / "postoperative_measurements.csv")
    preoperative_columns = set(preoperative_rows[0]) if preoperative_rows else set()
    duringoperative_columns = set(duringoperative_rows[0]) if duringoperative_rows else set()
    postoperative_columns = set(postoperative_rows[0]) if postoperative_rows else set()
    plots = output / "parameter_distributions"
    plots.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | int]] = []
    category_rows: list[dict[str, str | int]] = []
    for parameter in parameters:
        parameter_id = str(parameter["id"])
        parameter_phase = str(parameter["phase"])
        if parameter_phase == "Pre-operative":
            rows = preoperative_rows
            column = parameter_id if parameter_id in preoperative_columns else ""
            filename_phase = "preoperative"
        elif parameter_phase == "During-operative":
            rows = duringoperative_rows
            column = parameter_id if parameter_id in duringoperative_columns else ""
            filename_phase = "duringoperative"
        else:
            rows = postoperative_rows
            column = _measurement_column(parameter, postoperative_specs, postoperative_columns)
            filename_phase = "postoperative"
        values = [row.get(column, "") if column else "" for row in rows]
        numeric_expected = parameter.get("parameter_type") in {"numeric", "derived_numeric", "time_series_numeric"}
        plot_path = plots / f"{filename_phase}_{re.sub(r'[^A-Za-z0-9_-]+', '_', parameter_id)}.html"
        kind, numeric_values, categories = _plot_distribution(
            plot_path,
            f"{parameter.get('name', parameter_id)} — {parameter_phase}",
            values,
            bins,
            numeric_expected=numeric_expected,
            theme=theme,
        )
        observed_count = len(numeric_values) if numeric_expected else sum(bool(value.strip()) for value in values)
        summary: dict[str, str | int] = {
            "parameter_id": parameter_id,
            "parameter": str(parameter.get("name", parameter_id)),
            "phase": parameter_phase,
            "kind": kind,
            "patient_count": len(rows),
            "observed_count": observed_count,
            "missing_count": len(rows) - observed_count,
            "mean": "",
            "median": "",
            "std": "",
            "min": "",
            "max": "",
            "plot": str(plot_path.relative_to(output)),
        }
        if numeric_values:
            summary.update({
                "mean": _format(statistics.fmean(numeric_values)),
                "median": _format(statistics.median(numeric_values)),
                "std": _format(statistics.pstdev(numeric_values)),
                "min": _format(min(numeric_values)),
                "max": _format(max(numeric_values)),
            })
        summary_rows.append(summary)
        category_rows.extend(
            {
                "parameter_id": parameter_id,
                "phase": parameter_phase,
                "category": category,
                "patient_count": count,
                "percent_of_observed": _format(100.0 * count / observed_count),
            }
            for category, count in sorted(categories.items())
        )

    summary_fields = ["parameter_id", "parameter", "phase", "kind", "patient_count", "observed_count", "missing_count", "mean", "median", "std", "min", "max", "plot"]
    with (output / "parameter_distribution_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "parameter_distribution_categories.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["parameter_id", "phase", "category", "patient_count", "percent_of_observed"])
        writer.writeheader()
        writer.writerows(category_rows)
    return len(summary_rows)
