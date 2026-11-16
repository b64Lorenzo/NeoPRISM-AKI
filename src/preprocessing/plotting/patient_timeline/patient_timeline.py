"""Build an offline, pseudonymous dashboard for perioperative patient timelines."""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import quote
from zoneinfo import ZoneInfo

from src.plotlib.plotly_output import plotly_template, write_plotly_html
from src.preprocessing.plotting.patient_timeline.timeline_dashboard_view import (
    DashboardLayout,
    TimelineDashboardView,
)
from src.preprocessing.postoperative.processor import aggregate_measurement
from src.preprocessing.preoperative.processor import PreoperativeProcessor
from src.utils.csv_utils import PatientCsvReader, is_missing, normalise, number_component
from src.utils.patient_files import discover_csv_files, group_patient_files

PHASES = ("Pre-operative", "During-operative", "Post-operative")
UNCLASSIFIED_PHASE = "Boundary unavailable"
NUMERIC_TYPES = {"numeric", "ordinal", "time_series_numeric", "derived_numeric", "derived_duration"}


class PatientTimelineDashboard:
    """Generate a patient-selectable dashboard with an operation chart and phase tables."""

    def __init__(self, parameter_map: Mapping) -> None:
        self.parameter_map = parameter_map
        self.parameters = tuple(parameter_map.get("parameters", []))
        self.measurements = {
            item["id"]: item for item in parameter_map.get("postoperative_measurements", [])
        }
        self.missing_tokens = {
            normalise(token) for token in parameter_map.get("missing_value_tokens", [])
        }

    @staticmethod
    def _phase(timestamp: float, start: float | None, end: float | None) -> str:
        if start is None:
            return UNCLASSIFIED_PHASE
        if timestamp < start:
            return "Pre-operative"
        if end is None:
            return UNCLASSIFIED_PHASE
        return "During-operative" if timestamp < end else "Post-operative"

    @staticmethod
    def _value(raw_value: str, parameter: Mapping) -> str | None:
        if parameter.get("parameter_type") in NUMERIC_TYPES:
            numeric = number_component(raw_value)
            return format(numeric, ".12g") if numeric is not None and math.isfinite(numeric) else None
        return raw_value[:500]

    def generate(
        self,
        inputs: Iterable[str | Path],
        output: str | Path,
        *,
        patient_ids: Iterable[str] = (),
        compare_all_patients: bool = False,
        related_plots: Iterable[str | Path] = (),
        theme: str = "light",
        exclude_output_dir: str | Path | None = None,
        layout: DashboardLayout | None = None,
    ) -> dict:
        """Write the patient dashboard from configured class and event inventories."""
        destination = Path(output)
        if destination.suffix.casefold() != ".html":
            raise ValueError("Patient timeline dashboard output must use the .html suffix")
        if not isinstance(compare_all_patients, bool):
            raise ValueError("compare_all_patients must be boolean")
        plotly_template(theme)
        related_plot_refs = self._related_plot_references(related_plots, destination)
        families = group_patient_files(discover_csv_files(inputs, exclude_output_dir or destination))
        if not families:
            raise ValueError("No CSV patient families found for the timeline dashboard")
        available = [f"patient_{index:03d}" for index in range(1, len(families) + 1)]
        requested = list(dict.fromkeys(str(value) for value in patient_ids))
        if set(requested) - set(available):
            raise ValueError("Unknown patient selection in timeline dashboard")
        selected = requested or available
        patients = [
            self._patient_dashboard_data(patient_id, paths)
            for patient_id, paths in zip(available, families) if patient_id in selected
        ]
        try:
            import plotly.graph_objects as go
        except ImportError as error:
            raise RuntimeError(
                "Patient timeline dashboards require Plotly; install dependencies from requirements.txt"
            ) from error

        figure = go.Figure()
        total_events = total_data = total_series_points = 0
        for index, patient in enumerate(patients):
            chart_visible = compare_all_patients or index == 0
            total_events += patient["event_count"]
            total_data += patient["data_count"]
            total_series_points += patient["series_point_count"]
            for series in patient["operation_series"]:
                series_name = (
                    f"{patient['patient_id']} — {series['name']}"
                    if compare_all_patients else series["name"]
                )
                figure.add_trace(go.Scatter(
                    x=series["hours"], y=series["normalized"], customdata=series["values"],
                    mode="lines+markers", name=series_name, legendgroup=patient["patient_id"],
                    meta={"patient": patient["patient_id"], "series": series["name"]},
                    line={"width": 1.6}, marker={"size": 4}, visible=chart_visible,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>Elapsed time: %{x:.2f} h"
                        "<br>Raw value: %{customdata:.6g}"
                        "<br>Normalized value: %{y:.3f}<extra></extra>"
                    ),
                ))
        figure.update_xaxes(title_text="Elapsed hours from operation start (no calendar dates)", zeroline=True)
        figure.update_yaxes(title_text="Normalized value (0–1)", range=[-0.05, 1.05])
        layout = layout or DashboardLayout()
        title = (
            "Perioperative dashboard — all patients"
            if compare_all_patients else f"Perioperative dashboard — {patients[0]['patient_id']}"
        )
        figure.update_layout(
            title=title, showlegend=False,
            template=plotly_template(theme), hovermode="closest", height=layout.chart_height,
            margin={"l": 80, "r": 24, "t": 65, "b": 55},
        )
        cohort = {}
        if compare_all_patients:
            pre_headers, pre_rows = self._cohort_phase_table(patients, "pre_rows")
            post_headers, post_rows = self._cohort_phase_table(patients, "post_rows")
            summary_headers, summary_rows = self._cohort_summary_table(patients)
            cohort = {
                "pre": {"headers": pre_headers, "rows": pre_rows},
                "post": {"headers": post_headers, "rows": post_rows},
                "summary": {"headers": summary_headers, "rows": summary_rows},
            }
        view = TimelineDashboardView(patients, cohort, related_plot_refs, theme, layout, compare_all_patients)
        write_plotly_html(
            figure, destination,
            post_script=view.script(), head=view.head(),
            before_plot=view.before_plot(), after_plot=view.after_plot(),
        )
        return {
            "patient_count": len(patients), "event_count": total_events,
            "data_point_count": total_data,
            "operation_series_point_count": total_series_points,
            "timeline_point_count": total_events + total_data, "output": destination,
        }

    @staticmethod
    def _related_plot_references(related_plots: Iterable[str | Path], destination: Path) -> list[dict[str, str]]:
        output_path = destination.resolve()
        references = []
        seen = set()
        for value in related_plots:
            path = Path(value)
            if path.suffix.casefold() != ".html" or not path.is_file():
                continue
            resolved = path.resolve()
            if resolved == output_path or resolved in seen:
                continue
            try:
                relative = os.path.relpath(resolved, output_path.parent)
            except ValueError:
                continue
            seen.add(resolved)
            references.append({
                "label": path.stem,
                "src": quote(Path(relative).as_posix(), safe="/._-"),
            })
        return references

    @staticmethod
    def _cohort_phase_table(patients: list[dict], key: str) -> tuple[list[str], list[list[str]]]:
        row_labels = []
        patient_values = []
        for patient in patients:
            values = {}
            counts = Counter()
            for name, value, time in patient[key]:
                label = f"Event: {value}" if name == "Event" else name
                counts[label] += 1
                if counts[label] > 1:
                    label = f"{label} ({counts[label]})"
                if label not in row_labels:
                    row_labels.append(label)
                values[label] = f"{value} — {time}"
            patient_values.append(values)
        headers = ["Data or event", *(patient["patient_id"] for patient in patients)]
        rows = [[label, *(values.get(label, "—") for values in patient_values)] for label in row_labels]
        return headers, rows or [["No configured records", *("—" for _ in patients)]]

    @staticmethod
    def _cohort_summary_table(patients: list[dict]) -> tuple[list[str], list[list[str]]]:
        labels = list(dict.fromkeys(
            row[0] for patient in patients for row in patient["summary_rows"]
        ))
        headers = ["Patient", *labels]
        rows = []
        for patient in patients:
            values = {row[0]: row[1] for row in patient["summary_rows"]}
            rows.append([patient["patient_id"], *(values.get(label, "—") for label in labels)])
        return headers, rows

    @staticmethod
    def _parse_duration_datetime(raw: str, timezone_name: str) -> float | None:
        value = raw.strip()
        if not value or not ("T" in value.upper() or " " in value):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
        except (ValueError, KeyError):
            return None
        timestamp = parsed.timestamp()
        return timestamp if math.isfinite(timestamp) else None

    @classmethod
    def _duration_seconds(
        cls, reader: PatientCsvReader, duration: Mapping, boundary_times: Mapping[str, float | None],
    ) -> float | None:
        method = duration["method"]
        if method == "phase_boundary_interval":
            start = boundary_times.get(duration["start_boundary"])
            end = boundary_times.get(duration["end_boundary"])
            if start is None or end is None or end < start:
                return None
            return end - start if duration.get("aggregation") == "difference" else None

        window_start = boundary_times.get(duration["match_start_boundary"])
        window_end = boundary_times.get(duration["match_end_boundary"])
        if window_start is None or window_end is None or window_end < window_start:
            return None
        start_indexes = reader.columns.get(normalise(duration["start_field"]), [])
        end_indexes = reader.columns.get(normalise(duration["end_field"]), [])
        if len(start_indexes) != 1 or len(end_indexes) != 1:
            return None
        timezone_name = duration["timezone"]
        intervals = set()
        for row in reader.rows():
            if start_indexes[0] >= len(row) or end_indexes[0] >= len(row):
                continue
            start = cls._parse_duration_datetime(row[start_indexes[0]], timezone_name)
            end = cls._parse_duration_datetime(row[end_indexes[0]], timezone_name)
            if start is not None and end is not None and end >= start:
                intervals.add((start, end))
        if not intervals:
            return None
        if duration.get("aggregation") != "sum_overlaps":
            return None
        clipped = [
            (max(start, window_start), min(end, window_end))
            for start, end in intervals
            if min(end, window_end) > max(start, window_start)
        ]
        merged = []
        for start, end in sorted(clipped):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return sum(end - start for start, end in merged)

    def _configured_duration_rows(
        self, reader: PatientCsvReader, boundary_times: Mapping[str, float | None],
    ) -> list[tuple[str, str, str]]:
        units = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0}
        rows = []
        for measurement in self.measurements.values():
            duration = measurement.get("duration")
            if not duration:
                continue
            seconds = self._duration_seconds(reader, duration, boundary_times)
            if seconds is None or seconds < 0 or not math.isfinite(seconds):
                value = "Unavailable"
            else:
                value = f"{seconds / units[duration['unit']]:.{duration['precision']}f} {duration['unit']}"
            rows.append((
                duration["label"],
                value,
                "XML-configured duration; source datetimes are not shown",
            ))
        return rows

    def _patient_dashboard_data(self, patient_id: str, paths: tuple[Path, ...]) -> dict:
        boundaries = PreoperativeProcessor(dict(self.parameter_map)).process(paths)
        reader = PatientCsvReader(paths)
        start, end = boundaries.cutoff, boundaries.end_cutoff
        post_end = boundaries.postoperative_end_cutoff
        boundary_times = {
            "operation_start": start,
            "operation_end": end,
            "postoperative_end": post_end,
        }
        event_counts, data_counts = Counter(), Counter()
        pre_latest, data_by_parameter = {}, defaultdict(list)
        events_by_phase = {phase: [] for phase in (*PHASES, UNCLASSIFIED_PHASE)}
        series_values = defaultdict(list)
        unplaced_events = unplaced_data = 0

        boundary_config = self.parameter_map.get("phase_boundaries", {})
        event_fields = list(dict.fromkeys(
            field for boundary in boundary_config.values()
            for field in boundary.get("event_fields", [])
        ))
        event_indexes = list(dict.fromkeys(
            index for field in event_fields
            for index in reader.columns.get(normalise(field), [])
        ))
        event_index_set = set(event_indexes)
        death_indexes = {}
        for parameter_id in ("recorded_death_days", "mortality_30d"):
            parameter = next((item for item in self.parameters if item["id"] == parameter_id), None)
            if parameter is None:
                continue
            measurement = self.measurements.get(parameter_id, {})
            fields = measurement.get("source_fields", parameter.get("source_fields", []))
            death_indexes[parameter_id] = list(dict.fromkeys(
                index for field in fields
                for index in reader.columns.get(normalise(field), [])
                if len(reader.columns.get(normalise(field), [])) == 1
            ))
        death_raw_values = {parameter_id: [] for parameter_id in death_indexes}
        timeless_classes = {name.casefold() for name in self.parameter_map.get("timeless_class_names", [])}
        class_prefixes = tuple(
            f"{normalise(name)}." for name in self.parameter_map.get("during_class_names", [])
            if name.casefold() not in timeless_classes
        )
        series_indexes = {
            index: header for index, header in enumerate(reader.headers)
            if index not in event_index_set
            and len(reader.columns.get(normalise(header), [])) == 1
            and any(normalise(header).startswith(prefix) for prefix in class_prefixes)
        }
        parameter_indexes = {}
        for parameter in self.parameters:
            parameter_id = parameter["id"]
            measurement = self.measurements.get(parameter_id, {})
            if (
                parameter.get("timed") is False
                or measurement.get("record_scope") == "patient"
                or measurement.get("duration")
            ):
                parameter_indexes[parameter_id] = []
                continue
            fields = (measurement.get("source_fields", parameter.get("source_fields", []))
                      if parameter.get("phase") == "Post-operative"
                      else parameter.get("source_fields", []))
            parameter_indexes[parameter_id] = list(dict.fromkeys(
                index for field in fields
                for index in reader.columns.get(normalise(field), [])
                if len(reader.columns.get(normalise(field), [])) == 1
            ))
        names = {
            item["id"]: str(item.get("clinical_parameter") or item.get("name") or item["id"])
            for item in self.parameters
        }

        for row in reader.rows():
            timestamp = reader.timestamp(row)
            phase = self._phase(timestamp, start, end) if timestamp is not None else UNCLASSIFIED_PHASE
            hours = (timestamp - start) / 3600.0 if timestamp is not None and start is not None else None
            for parameter_id, indexes in death_indexes.items():
                if parameter_id == "mortality_30d" and phase != "Post-operative":
                    continue
                for index in indexes:
                    raw = row[index].strip() if index < len(row) else ""
                    if raw and not is_missing(raw, self.missing_tokens):
                        death_raw_values[parameter_id].append(raw)
            values = []
            for index in event_indexes:
                if index < len(row):
                    value = row[index].strip()
                    if value and not is_missing(value, self.missing_tokens) and value not in values:
                        values.append(value)
            row_events = []
            for value in values:
                if hours is None:
                    unplaced_events += 1
                    continue
                event = {"phase": phase, "hours": hours, "value": value, "details": []}
                event_counts[phase] += 1
                events_by_phase[phase].append(event)
                row_events.append(event)

            if phase == "During-operative" and hours is not None:
                for index, name in series_indexes.items():
                    if index >= len(row):
                        continue
                    raw = row[index].strip()
                    numeric = None if is_missing(raw, self.missing_tokens) else number_component(raw)
                    if numeric is not None and math.isfinite(numeric):
                        series_values[name].append((hours, numeric))

            row_details = []
            for parameter in self.parameters:
                parameter_id = parameter["id"]
                indexes = parameter_indexes.get(parameter_id, [])
                expected = parameter.get("phase")
                if not indexes:
                    continue
                if phase == UNCLASSIFIED_PHASE:
                    for index in indexes:
                        raw = row[index].strip() if index < len(row) else ""
                        if not is_missing(raw, self.missing_tokens) and self._value(raw, parameter) is not None:
                            unplaced_data += 1
                    continue
                if phase != expected:
                    continue
                measurement = self.measurements.get(parameter_id, {})
                window = measurement.get("window_hours")
                if expected == "Post-operative" and end is not None and window is not None:
                    if timestamp > end + float(window) * 3600.0:
                        continue
                for index in indexes:
                    if index >= len(row):
                        continue
                    raw = row[index].strip()
                    if is_missing(raw, self.missing_tokens):
                        continue
                    value = self._value(raw, parameter)
                    if value is None:
                        continue
                    point = {"hours": hours, "timestamp": timestamp, "value": value}
                    row_details.append({
                        "parameter_id": parameter_id,
                        "name": names[parameter_id],
                        "value": value,
                        "unit": str(parameter.get("unit") or "").strip(),
                        "phase": phase,
                    })
                    if expected == "Pre-operative" and parameter.get("parameter_type") != "time_series_numeric":
                        previous = pre_latest.get(parameter_id)
                        if previous is None or timestamp >= previous["timestamp"]:
                            pre_latest[parameter_id] = point
                    else:
                        data_counts[phase] += 1
                        data_by_parameter[(phase, parameter_id)].append(point)
            for event in row_events:
                event["details"] = row_details

        data_counts["Pre-operative"] += len(pre_latest)
        pre_rows = []
        demographics = []
        for parameter in self.parameters:
            if parameter.get("phase") != "Pre-operative":
                continue
            parameter_id = parameter["id"]
            if parameter.get("timed") is False:
                value = str(boundaries.features.get(parameter_id) or "").strip()
                unit = str(parameter.get("unit") or "").strip()
                formatted = f"{value}{f' {unit}' if unit else ''}" if value else "Unavailable"
                demographics.append((names[parameter_id], formatted))
                continue
            observations = data_by_parameter.get(("Pre-operative", parameter_id), [])
            point = pre_latest.get(parameter_id)
            if point is None and observations:
                point = max(observations, key=lambda item: item["timestamp"])
            unit = str(parameter.get("unit") or "").strip()
            if point is not None:
                value = f"{point['value']}{f' {unit}' if unit else ''}"
                if observations:
                    value += f" ({len(observations)} value(s))"
                time = self._dashboard_time(point["hours"])
            else:
                fallback = boundaries.features.get(parameter_id, "")
                if fallback is not None and str(fallback).strip():
                    value = f"{fallback}{f' {unit}' if unit else ''}"
                    time = "Static/cutoff value"
                else:
                    value = boundaries.feature_status.get(parameter_id, "no_preoperative_value").replace("_", " ")
                    time = "—"
            pre_rows.append((names[parameter_id], value, time))
        pre_rows.extend(
            ("Event", item["value"], self._dashboard_time(item["hours"]))
            for item in sorted(events_by_phase["Pre-operative"], key=lambda item: item["hours"])
        )

        post_rows = []
        for parameter in self.parameters:
            if parameter.get("phase") != "Post-operative":
                continue
            parameter_id = parameter["id"]
            observations = data_by_parameter.get(("Post-operative", parameter_id), [])
            unit = str(parameter.get("unit") or "").strip()
            if observations:
                latest = max(observations, key=lambda item: item["timestamp"])
                value = f"{latest['value']}{f' {unit}' if unit else ''} ({len(observations)} value(s))"
                time = self._dashboard_time(latest["hours"])
            else:
                measurement = self.measurements.get(parameter_id, {})
                if measurement.get("record_scope") == "patient":
                    value = "Patient-scope result not present in event CSV"
                else:
                    value = "No value recorded in configured window"
                time = "—"
            post_rows.append((names[parameter_id], value, time))
        post_rows.extend(
            ("Event", item["value"], self._dashboard_time(item["hours"]))
            for item in sorted(events_by_phase["Post-operative"], key=lambda item: item["hours"])
        )

        timeline_events = sorted(
            (event for phase in (*PHASES, UNCLASSIFIED_PHASE) for event in events_by_phase[phase]),
            key=lambda item: item["hours"],
        )
        series = []
        for name, observations in sorted(series_values.items()):
            observations.sort(key=lambda item: item[0])
            values = [value for _, value in observations]
            low, high = min(values), max(values)
            normalized = [0.5] * len(values) if high == low else [(v - low) / (high - low) for v in values]
            series.append({"name": name, "hours": [h for h, _ in observations],
                           "values": values, "normalized": normalized})
        duration = (end - start) / 3600.0 if start is not None and end is not None else None
        sample_count = sum(len(item["values"]) for item in series)
        death_days = aggregate_measurement(
            death_raw_values.get("recorded_death_days", []),
            self.measurements.get(
                "recorded_death_days", {"aggregation": "max", "value_type": "numeric"}
            ),
        )
        if death_days is not None and (not math.isfinite(death_days) or death_days < 0):
            death_days = None
        mortality_flag = aggregate_measurement(
            death_raw_values.get("mortality_30d", []),
            self.measurements.get(
                "mortality_30d", {"aggregation": "any_true", "value_type": "boolean"}
            ),
        )
        if death_days is not None or mortality_flag == 1.0:
            death_status = "Yes (recorded)"
        elif mortality_flag == 0.0:
            death_status = "No death indicated by available mortality flag"
        else:
            death_status = "Not recorded"
        death_distance = f"{format(death_days, '.12g')} days" if death_days is not None else "Unavailable"
        milestones = [
            {"name": name, "hours": (when - start) / 3600 if when is not None and start is not None else None}
            for name, when in (
                ("Operation start", start),
                ("Operation end", end),
                ("Post-operative section end", post_end),
            )
        ]
        summary = [
            ("Operation start boundary", "Found" if start is not None else "Not found", "—"),
            ("Operation end boundary", "Found" if end is not None else "Not found", "—"),
            ("Post-operative section end boundary", "Found" if post_end is not None else "Not found", "Configured event marker"),
            ("Operation duration", f"{duration:.2f} h" if duration is not None else "Unavailable", "—"),
            ("Death status", death_status, "Configured evidence only"),
            ("Distance from operation to death", death_distance,
             "Recorded value; not timestamp-derived" if death_days is not None else "No valid day value"),
            ("Configured during-operative classes", str(len(self.parameter_map.get("during_class_names", []))), "XML inventory"),
            ("Numeric time-series fields plotted", str(len(series)), "During operation only"),
            ("Numeric time-series samples", str(sample_count), "During operation only"),
            ("Events / mapped values", f"{sum(event_counts.values())} / {sum(data_counts.values())}", "Timed records"),
            ("Boundary-unavailable events", str(event_counts[UNCLASSIFIED_PHASE]), "—"),
            ("Unplaced records", f"{unplaced_events} event(s), {unplaced_data} data value(s)", "No usable timestamp"),
        ]
        duration_rows = self._configured_duration_rows(reader, boundary_times)
        summary.extend(duration_rows)
        summary.extend(
            (f"{phase} events / mapped values", f"{event_counts[phase]} / {data_counts[phase]}", "—")
            for phase in PHASES
        )
        return {
            "patient_id": patient_id, "operation_series": series,
            "timeline_events": timeline_events, "pre_rows": pre_rows,
            "demographics": demographics, "milestones": milestones,
            "duration_stats": [(label, value) for label, value, _ in duration_rows],
            "post_rows": post_rows, "summary_rows": summary,
            "event_count": sum(event_counts.values()),
            "data_count": sum(data_counts.values()), "series_point_count": sample_count,
        }

    @staticmethod
    def _dashboard_time(hours: float | None) -> str:
        return "Operation start unavailable" if hours is None else f"{hours:+.2f} h from operation start"
