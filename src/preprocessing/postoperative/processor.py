"""Aggregation helpers for mapped postoperative measures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.postoperative.outcome_rules import OutcomeClassifier
from src.preprocessing.preoperative.processor import PreoperativeResult
from src.utils.csv_utils import PatientCsvReader, is_missing, normalise, number, number_component

POSTOPERATIVE_PHASE = "Post-operative"


def _boolean(value: object) -> bool | None:
    token = str(value).strip().casefold()
    if token in {"1", "1.0", "yes", "true", "y", "dead", "deceased", "died"}:
        return True
    if token in {"0", "0.0", "no", "false", "n", "alive"}:
        return False
    numeric = number(token)
    if numeric == 1:
        return True
    if numeric == 0:
        return False
    return None


def aggregate_measurement(values: Iterable[object], spec: Mapping) -> float | None:
    """Aggregate observed values using an explicit map-configured strategy."""
    value_type = spec.get("value_type", "numeric")
    if value_type == "boolean":
        parsed = [value for value in (_boolean(item) for item in values) if value is not None]
        numeric = [1.0 if value else 0.0 for value in parsed]
    else:
        numeric = [
            value for value in (number_component(item) for item in values) if value is not None
        ]
    if not numeric:
        return None

    aggregation = spec.get("aggregation", "last")
    if aggregation == "first":
        return numeric[0]
    if aggregation == "last":
        return numeric[-1]
    if aggregation == "min":
        return min(numeric)
    if aggregation == "max":
        return max(numeric)
    if aggregation == "mean":
        return sum(numeric) / len(numeric)
    if aggregation == "sum":
        return sum(numeric)
    if aggregation == "any_true" and value_type == "boolean":
        return float(any(numeric))
    raise ValueError(f"Unsupported postoperative aggregation: {aggregation}")


def _measurement_text(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if value.is_integer() else format(value, ".12g")


@dataclass(frozen=True)
class PostoperativeResult:
    measurements: dict[str, str]
    target: int | None
    target_status: str
    label_basis: str


class PostoperativeProcessor:
    """Aggregate postoperative parameters and apply configured binary criteria."""

    def __init__(
        self,
        parameter_map: dict,
        *,
        outcome_column: str | None = None,
        success_values: Iterable[str] | None = None,
        failure_values: Iterable[str] | None = None,
    ) -> None:
        self.parameter_map = parameter_map
        self.parameter_specs = tuple(
            parameter
            for parameter in parameter_map["parameters"]
            if parameter["phase"] == POSTOPERATIVE_PHASE
        )
        self.parameters = tuple(parameter["id"] for parameter in self.parameter_specs)
        self.measurement_specs = tuple(parameter_map.get("postoperative_measurements", []))
        self.post_class_failures = tuple(parameter_map.get("post_class_failures", []))
        if any(
            spec.get("record_scope", "postoperative") not in {"postoperative", "patient"}
            for spec in self.measurement_specs
        ):
            raise ValueError("Unsupported record_scope in postoperative measurements")
        if any(
            spec.get("record_scope") == "patient" and spec.get("window_hours") is not None
            for spec in self.measurement_specs
        ):
            raise ValueError("patient record_scope cannot have window_hours")
        self.outcome_column = outcome_column
        self.classifier = OutcomeClassifier(
            parameter_map,
            success_values=success_values,
            failure_values=failure_values,
        )
        self.missing_tokens = {
            normalise(token) for token in parameter_map.get("missing_value_tokens", [])
        }

    def process(
        self,
        paths: str | Path | Iterable[str | Path],
        preoperative: PreoperativeResult,
    ) -> PostoperativeResult:
        reader = PatientCsvReader(paths)
        rules = self.parameter_map.get("patient_outcome_rules", {})
        post_start = preoperative.end_cutoff
        direct_columns = list(rules.get("direct_outcome_columns", []))
        if self.outcome_column:
            direct_columns.insert(0, self.outcome_column)
        direct_indexes = [
            index
            for name in dict.fromkeys(direct_columns)
            for index in reader.columns.get(normalise(name), [])
        ]
        measurement_indexes = {
            spec["id"]: [
                index
                for name in spec.get("source_fields", [])
                for index in reader.columns.get(normalise(name), [])
            ]
            for spec in self.measurement_specs
        }
        scopes = {spec["id"]: spec for spec in self.measurement_specs}
        class_indexes = {
            name: [
                (
                    index,
                    scopes.get(parameter["id"], {}).get("record_scope") == "patient",
                    scopes.get(parameter["id"], {}).get("window_hours"),
                )
                for parameter in self.parameter_specs
                for field in parameter.get("source_fields", [])
                if field.startswith(name + ".") and len(field) > len(name) + 1
                for index in reader.columns.get(normalise(field), [])
            ]
            for name in self.post_class_failures
        }
        matched_classes: set[str] = set()
        direct_outcomes: list[str] = []
        raw_measurements: dict[str, list[str]] = {key: [] for key in measurement_indexes}

        for row in reader.rows():
            timestamp = reader.timestamp(row)
            postoperative_row = (
                post_start is not None and timestamp is not None and timestamp >= post_start
            )
            for name, indexes in class_indexes.items():
                if any(
                    index < len(row)
                    and (patient_scope or postoperative_row)
                    and (
                        hours is None
                        or (
                            post_start is not None
                            and timestamp is not None
                            and timestamp <= post_start + float(hours) * 3600.0
                        )
                    )
                    and bool(row[index].strip())
                    and not is_missing(row[index], self.missing_tokens)
                    for index, patient_scope, hours in indexes
                ):
                    matched_classes.add(name)
            if postoperative_row:
                direct_outcomes.extend(
                    row[index].strip()
                    for index in direct_indexes
                    if index < len(row) and not is_missing(row[index], self.missing_tokens)
                )

            for spec in self.measurement_specs:
                patient_scope = spec.get("record_scope") == "patient"
                if not patient_scope and not postoperative_row:
                    continue
                indexes = measurement_indexes.get(spec["id"], [])
                hours = spec.get("window_hours")
                if not indexes or (
                    hours is not None
                    and (
                        post_start is None
                        or timestamp is None
                        or timestamp > post_start + float(hours) * 3600.0
                    )
                ):
                    continue
                raw_measurements[spec["id"]].extend(
                    row[index].strip()
                    for index in indexes
                    if index < len(row) and not is_missing(row[index], self.missing_tokens)
                )

        measurements: dict[str, str] = {}
        for spec in self.measurement_specs:
            value = aggregate_measurement(raw_measurements.get(spec["id"], []), spec)
            measurements[spec["id"]] = _measurement_text(value)

        outcome = self.classifier.classify(measurements, categories=direct_outcomes)
        if matched_classes:
            return PostoperativeResult(
                measurements, -1, "failure", ", ".join(sorted(matched_classes))
            )
        basis = (
            outcome.label_basis
            or "No configured outcome criterion matched available postoperative evidence."
        )
        return PostoperativeResult(
            measurements,
            outcome.target,
            outcome.status,
            basis,
        )
