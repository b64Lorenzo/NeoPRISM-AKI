"""Export selected time-series readings with clocks relative to phase boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.preprocessing.preoperative.processor import PreoperativeResult
from src.utils.csv_utils import PatientCsvReader, is_missing, normalise, number_component


class TimeSeriesProcessor:
    """Keep raw time-series readings separate from patient features and outcomes."""

    def __init__(self, parameter_map: dict) -> None:
        self.enabled = parameter_map.get("export_time_series", False)
        measurements = {
            item["id"]: item for item in parameter_map.get("postoperative_measurements", [])
        }
        self.parameters = []
        for item in parameter_map["parameters"]:
            if item.get("parameter_type") != "time_series_numeric":
                continue
            phase = item.get("phase")
            if phase not in {"During-operative", "Post-operative"}:
                continue
            measurement = measurements.get(item["id"], {}) if phase == "Post-operative" else {}
            if measurement.get("record_scope") == "patient":
                continue
            self.parameters.append(
                (
                    phase,
                    item["id"],
                    measurement.get("source_fields", item.get("source_fields", [])),
                    measurement.get("window_hours"),
                )
            )
        self.parameters.sort(key=lambda item: item[0] != "During-operative")
        self.missing_tokens = {
            normalise(token) for token in parameter_map.get("missing_value_tokens", [])
        }

    def process(
        self,
        paths: str | Path | Iterable[str | Path],
        boundaries: PreoperativeResult,
    ) -> list[dict[str, str]]:
        if not self.enabled or not self.parameters or boundaries.end_cutoff is None:
            return []
        reader = PatientCsvReader(paths)
        indexes = [
            list(
                dict.fromkeys(
                    index
                    for field in fields
                    for index in reader.columns.get(normalise(field), [])
                    if len(reader.columns.get(normalise(field), [])) == 1
                )
            )
            for _, _, fields, _ in self.parameters
        ]
        readings: list[tuple[int, float, int, dict[str, str]]] = []
        for row_number, row in enumerate(reader.rows()):
            timestamp = reader.timestamp(row)
            if timestamp is None:
                continue
            for order, ((phase, parameter_id, _, window_hours), columns) in enumerate(
                zip(self.parameters, indexes)
            ):
                start = boundaries.cutoff if phase == "During-operative" else boundaries.end_cutoff
                if start is None or timestamp < start:
                    continue
                if phase == "During-operative" and timestamp >= boundaries.end_cutoff:
                    continue
                if window_hours is not None and timestamp > start + float(window_hours) * 3600:
                    continue
                for index in columns:
                    if index >= len(row) or is_missing(row[index], self.missing_tokens):
                        continue
                    value = number_component(row[index])
                    if value is None:
                        continue
                    readings.append(
                        (
                            order,
                            timestamp,
                            row_number,
                            {
                                "phase": phase,
                                "parameter_id": parameter_id,
                                "t_seconds": format(timestamp - start, ".12g"),
                                "value": format(value, ".12g"),
                            },
                        )
                    )
        readings.sort(key=lambda item: (item[0], item[1], item[2]))
        return [reading for _, _, _, reading in readings]
