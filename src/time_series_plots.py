"""Plot configured time-series parameters for individual or compared patients."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping

from src.csv_utils import number_component
from src.plotly_output import plotly_template, write_plotly_html


TIME_SERIES_PHASES = {"During-operative", "Post-operative"}
REQUIRED_COLUMNS = {"patient_id", "phase", "parameter_id", "t_seconds", "value"}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "plot"


class TimeSeriesPlotter:
    """Render one chart per parameter or one multi-panel chart per class."""

    def __init__(self, parameter_map: Mapping) -> None:
        self.parameters = {
            str(item["id"]): item
            for item in parameter_map.get("parameters", [])
            if item.get("phase") in TIME_SERIES_PHASES
            and item.get("parameter_type") == "time_series_numeric"
        }

    def generate(
        self,
        input_csv: str | Path,
        output_dir: str | Path,
        *,
        parameter_ids: Iterable[str] = (),
        class_names: Iterable[str] = (),
        patient_ids: Iterable[str] = (),
        group_by: str = "parameter",
        theme: str = "light",
    ) -> dict:
        """Plot every selected observation and return aggregate artifact counts."""
        if group_by not in {"parameter", "class"}:
            raise ValueError("group_by must be parameter or class")
        parameters = self._select_parameters(parameter_ids, class_names)
        rows = self._read_rows(Path(input_csv))
        available_patients = {row["patient_id"].strip() for row in rows if row.get("patient_id", "").strip()}
        selected_patients = list(dict.fromkeys(patient_ids))
        unknown_patients = set(selected_patients) - available_patients
        if unknown_patients:
            raise ValueError("Unknown patient selection in time-series CSV")
        if not selected_patients:
            selected_patients = sorted(available_patients)

        selected_patient_set = set(selected_patients)
        observations: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            patient_id = row.get("patient_id", "").strip()
            parameter_id = row.get("parameter_id", "").strip()
            parameter = parameters.get(parameter_id)
            if parameter is None or patient_id not in selected_patient_set:
                continue
            if row.get("phase", "").strip() != parameter.get("phase"):
                raise ValueError("Time-series phase does not match its configured parameter")
            timestamp = number_component(row.get("t_seconds", ""))
            value = number_component(row.get("value", ""))
            if timestamp is None or value is None or not math.isfinite(timestamp) or not math.isfinite(value):
                raise ValueError("Time-series CSV contains a nonnumeric observation")
            if timestamp < 0:
                raise ValueError("Time-series elapsed seconds cannot be negative")
            observations[parameter_id][patient_id].append((timestamp, value))

        parameters = {
            parameter_id: parameter
            for parameter_id, parameter in parameters.items()
            if parameter_id in observations
        }
        if not parameters:
            raise ValueError("No time-series observations match the selected parameters and patients")
        for patients in observations.values():
            for values in patients.values():
                values.sort(key=lambda item: item[0])

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as error:
            raise RuntimeError(
                "Time-series plots require Plotly; install dependencies from requirements.txt"
            ) from error

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        groups: dict[str, list[str]] = defaultdict(list)
        if group_by == "parameter":
            for parameter_id in parameters:
                groups[parameter_id].append(parameter_id)
        else:
            for parameter_id, parameter in parameters.items():
                groups[str(parameter.get("class_name") or "unclassified")].append(parameter_id)

        plot_paths = []
        observation_count = 0
        plotted_patients = set()
        for group_name, parameter_group in groups.items():
            if group_by == "parameter":
                figure = go.Figure()
                filename = f"{_slug(group_name)}.html"
            else:
                figure = make_subplots(
                    rows=len(parameter_group),
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=min(0.08, 0.4 / max(1, len(parameter_group))),
                    subplot_titles=[
                        str(parameters[parameter_id].get("name") or parameter_id)
                        for parameter_id in parameter_group
                    ],
                )
                filename = f"{_slug(group_name)}.html"

            shown_legend = set()
            for row_index, parameter_id in enumerate(parameter_group, start=1):
                parameter = parameters[parameter_id]
                patient_series = observations[parameter_id]
                parameter_observations = sum(len(values) for values in patient_series.values())
                observation_count += parameter_observations
                plotted_patients.update(patient_series)
                for patient_id in selected_patients:
                    values = patient_series.get(patient_id)
                    if not values:
                        continue
                    show_legend = (
                        len(patient_series) > 1
                        and patient_id not in shown_legend
                    )
                    if show_legend:
                        shown_legend.add(patient_id)
                    trace = go.Scatter(
                        x=[point[0] / 3600.0 for point in values],
                        y=[point[1] for point in values],
                        mode="lines+markers",
                        name=patient_id,
                        legendgroup=patient_id,
                        showlegend=show_legend,
                        hovertemplate=(
                            "Elapsed: %{x:g} h<br>Value: %{y:g}<extra>%{fullData.name}</extra>"
                        ),
                    )
                    if group_by == "parameter":
                        figure.add_trace(trace)
                    else:
                        figure.add_trace(trace, row=row_index, col=1)

                title = str(parameter.get("name") or parameter.get("clinical_parameter") or parameter_id)
                if group_by == "parameter":
                    figure.update_layout(
                        title=f"{title} — {parameter.get('phase', '')}",
                        xaxis_title="Elapsed time (hours)",
                        yaxis_title=str(parameter.get("unit") or "Value"),
                    )
                else:
                    figure.update_yaxes(
                        title_text=str(parameter.get("unit") or "Value"),
                        row=row_index,
                        col=1,
                    )

            if group_by == "class":
                figure.update_layout(
                    title=(group_name if group_name != "unclassified" else "Unclassified parameters"),
                    height=max(420, 340 * len(parameter_group)),
                )
                figure.update_xaxes(title_text="Elapsed time (hours)", row=len(parameter_group), col=1)
            figure.update_layout(
                template=plotly_template(theme),
                hovermode="x unified",
                legend_title_text="Patient",
                margin={"l": 80, "r": 30, "t": 90, "b": 80},
            )
            path = output / filename
            write_plotly_html(figure, path)
            plot_paths.append(path)

        return {
            "plot_count": len(plot_paths),
            "parameter_count": len(parameters),
            "patient_count": len(plotted_patients),
            "observation_count": observation_count,
            "plots": plot_paths,
        }

    def _select_parameters(
        self,
        parameter_ids: Iterable[str],
        class_names: Iterable[str],
    ) -> dict[str, Mapping]:
        requested_ids = list(dict.fromkeys(parameter_ids))
        requested_classes = list(dict.fromkeys(class_names))
        unknown_ids = set(requested_ids) - self.parameters.keys()
        if unknown_ids:
            raise ValueError("Unknown or non-time-series parameter selection")
        available_classes = {
            str(parameter["class_name"])
            for parameter in self.parameters.values()
            if parameter.get("class_name")
        }
        unknown_classes = set(requested_classes) - available_classes
        if unknown_classes:
            raise ValueError("Unknown time-series class selection")
        return {
            parameter_id: parameter
            for parameter_id, parameter in self.parameters.items()
            if (not requested_ids or parameter_id in requested_ids)
            and (not requested_classes or parameter.get("class_name") in requested_classes)
        }

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or not REQUIRED_COLUMNS.issubset(fields):
                raise ValueError("Time-series CSV is missing required columns")
            return list(reader)
