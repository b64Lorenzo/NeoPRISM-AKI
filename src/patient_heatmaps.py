"""Rendering helpers kept separate from extraction and outcome rules."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.csv_utils import number_component
from src.plotly_output import plotly_template, write_plotly_html

HEATMAP_SAMPLE_LIMIT = 500


def _number(value: object) -> float | None:
    return number_component(value)


def _plotly_graph_objects():
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError("Heatmaps require Plotly; install dependencies from requirements.txt") from error
    return go


def write_placeholder_heatmap(path: Path, title: str, message: str, *, theme: str = "light") -> None:
    figure = _plotly_graph_objects().Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.update_layout(
        title=title,
        template=plotly_template(theme),
        height=350,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    write_plotly_html(figure, path)


def write_heatmap(
    path: Path,
    matrix: list[list[float]],
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    colorbar_label: str,
    row_axis_label: str = "Patient pseudonym",
    *,
    colorscale="Viridis",
    colorbar_tickvals: Sequence[float] = (),
    colorbar_ticktext: Sequence[str] = (),
    theme: str = "light",
) -> None:
    if not matrix or not column_labels:
        write_placeholder_heatmap(path, title, "No mapped measurements available to plot.", theme=theme)
        return

    colorbar = {"title": {"text": colorbar_label}}
    if colorbar_tickvals:
        colorbar["tickvals"] = list(colorbar_tickvals)
        colorbar["ticktext"] = list(colorbar_ticktext)
    figure = _plotly_graph_objects().Figure(
        data=[
            _plotly_graph_objects().Heatmap(
                z=matrix,
                x=column_labels,
                y=row_labels,
                zmin=0,
                zmax=1,
                colorscale=colorscale,
                colorbar=colorbar,
                hoverongaps=False,
                hovertemplate=(
                    "Patient/record: %{y}<br>Parameter: %{x}<br>Value: %{z:.3f}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title=title,
        template=plotly_template(theme),
        width=max(900, min(1800, 110 * len(column_labels) + 350)),
        height=max(400, min(1800, 24 * len(row_labels) + 250)),
        plot_bgcolor="#333333" if theme == "dark" else "#b8b8b8",
        margin={"l": 120, "r": 50, "t": 90, "b": 140},
    )
    figure.update_xaxes(title_text="Parameter", tickangle=-45, automargin=True)
    figure.update_yaxes(title_text=row_axis_label, autorange="reversed", automargin=True)
    if len(row_labels) > 80:
        step = max(1, len(row_labels) // 20)
        ticks = row_labels[::step]
        figure.update_yaxes(tickmode="array", tickvals=ticks, ticktext=ticks)
    write_plotly_html(figure, path)


def sample_patients(patients: list[dict], limit: int = HEATMAP_SAMPLE_LIMIT) -> list[dict]:
    return patients if len(patients) <= limit else random.Random(0).sample(patients, limit)


def build_parameter_matrix(patients: list[dict], parameters: Sequence[Mapping], *, value_key: str, categorical_ids: Iterable[str] = ()) -> tuple[list[list[float]], list[str]]:
    """Build a visualization matrix while retaining configured empty columns."""
    categorical = set(categorical_ids)
    matrix: list[list[float]] = [[] for _ in patients]
    labels: list[str] = []

    for parameter in parameters:
        parameter_id = str(parameter["id"])
        values = [patient.get(value_key, {}).get(parameter_id, "") for patient in patients]
        if parameter_id in categorical:
            categories = sorted({str(value) for value in values if value not in (None, "")})
            if any(value in (None, "") for value in values):
                categories.append("__MISSING__")
            for category in categories:
                label = "NA" if category == "__MISSING__" else category
                labels.append(f"{parameter_id}={label}")
                for row, value in zip(matrix, values):
                    row.append(1.0 if (str(value) == category or (category == "__MISSING__" and value in (None, ""))) else 0.0)
            continue

        numeric = [_number(value) for value in values]
        usable = [value for value in numeric if value is not None]
        if not usable:
            labels.append(f"{parameter_id} (no observed values)")
            for row in matrix:
                row.append(math.nan)
            continue
        low, high = min(usable), max(usable)
        labels.append(f"{parameter_id} (min-max)")
        for row, value in zip(matrix, numeric):
            row.append(math.nan if value is None else (0.5 if high == low else (value - low) / (high - low)))
    return matrix, labels


def write_parameter_heatmap(
    path: Path,
    patients: list[dict],
    parameters: Sequence[Mapping],
    *,
    value_key: str,
    title: str,
    categorical_ids: Iterable[str] = (),
    theme: str = "light",
) -> None:
    selected = sample_patients(patients)
    if not selected:
        write_placeholder_heatmap(path, title, "No patient records available to plot.", theme=theme)
        return
    matrix, labels = build_parameter_matrix(
        selected,
        parameters,
        value_key=value_key,
        categorical_ids=categorical_ids,
    )
    write_heatmap(
        path,
        matrix,
        [patient["patient_id"] for patient in selected],
        labels,
        title,
        "Min-max scaled value / one-hot indicator",
        theme=theme,
    )
