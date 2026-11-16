"""Read explicitly configured during-operative fields between the two event markers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.preprocessing.preoperative.processor import PreoperativeResult
from src.utils.csv_utils import PatientCsvReader, is_missing, normalise, number_component

DURING_PHASE = "During-operative"


class DuringoperativeProcessor:
    def __init__(self, parameter_map: dict) -> None:
        self.parameters = tuple(
            item for item in parameter_map["parameters"] if item["phase"] == DURING_PHASE
        )
        self.missing_tokens = {normalise(token) for token in parameter_map.get("missing_value_tokens", [])}

    def process(self, paths: str | Path | Iterable[str | Path], boundaries: PreoperativeResult) -> dict[str, str]:
        result = {item["id"]: "" for item in self.parameters}
        if not self.parameters or boundaries.cutoff is None or boundaries.end_cutoff is None:
            return result
        reader = PatientCsvReader(paths)
        indexes = {
            item["id"]: list(dict.fromkeys(
                index for field in item.get("source_fields", [])
                for index in reader.columns.get(normalise(field), [])
                if len(reader.columns.get(normalise(field), [])) == 1
            ))
            for item in self.parameters
        }
        best: dict[str, tuple[float, str]] = {}
        for row in reader.rows():
            timestamp = reader.timestamp(row)
            if timestamp is None or not boundaries.cutoff <= timestamp < boundaries.end_cutoff:
                continue
            for item in self.parameters:
                for index in indexes[item["id"]]:
                    if index >= len(row) or is_missing(row[index], self.missing_tokens):
                        continue
                    value = row[index].strip()
                    if item.get("parameter_type") in {"numeric", "derived_numeric", "time_series_numeric"}:
                        numeric = number_component(value)
                        if numeric is None:
                            continue
                        value = format(numeric, ".12g")
                    if item["id"] not in best or timestamp >= best[item["id"]][0]:
                        best[item["id"]] = (timestamp, value)
        for parameter_id, (_, value) in best.items():
            result[parameter_id] = value
        return result
