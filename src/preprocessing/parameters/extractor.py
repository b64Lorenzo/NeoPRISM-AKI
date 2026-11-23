
"""Phase-aware, per-patient parameter extraction.

Supports parameters belonging to multiple clinical phases.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.preprocessing.parameters.config import (
    PHASES,
    ExtractionConfig,
    load_config,
)
from src.utils.csv_utils import PatientCsvReader
from src.utils.patient_files import (
    discover_csv_files,
    group_patient_files,
    patient_family,
)


OUTPUT_FIELDS = [
    "time_minutes",
    "parameter_id",
    "value",
    "unit",
    "type",
    "source_field",
    "value_origin",
]

TRUE_VALUES = {"true", "yes", "1", "y"}
FALSE_VALUES = {"false", "no", "0", "n"}


def _normalise(value: object) -> str:
    return " ".join(
        str(value).strip().casefold().split()
    )


def _parse_value(
    raw: str,
    kind: str,
) -> str | None:
    """Parse and validate a configured parameter value."""
    value = raw.strip()

    if kind in {"numeric", "timeseries"}:
        component = value.split("|", 1)[0].strip()
        component = component.replace(",", ".")

        try:
            number = float(component)
        except ValueError:
            return None

        if not math.isfinite(number):
            return None

        return format(number, ".12g")

    if kind == "bool":
        token = _normalise(value)

        if token in TRUE_VALUES:
            return "true"

        if token in FALSE_VALUES:
            return "false"

        return None

    if kind == "str":
        return value or None

    raise ValueError(
        f"Unsupported parameter type: {kind}"
    )


def _timestamp(
    raw: str,
) -> tuple[float, str] | None:
    """Parse Unix seconds, milliseconds or ISO timestamps."""
    value = raw.strip()

    if not value:
        return None

    try:
        numeric = float(value)
    except ValueError:
        numeric = None

    if numeric is not None:
        if not math.isfinite(numeric):
            return None

        seconds = (
            numeric / 1000.0
            if abs(numeric) >= 100_000_000_000
            else numeric
        )

        try:
            dt = datetime.fromtimestamp(
                seconds,
                tz=timezone.utc,
            )
        except (
            OverflowError,
            OSError,
            ValueError,
        ):
            return None

        return (
            seconds,
            dt.isoformat().replace("+00:00", "Z"),
        )

    try:
        dt = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dt = dt.astimezone(timezone.utc)

    return (
        dt.timestamp(),
        dt.isoformat().replace("+00:00", "Z"),
    )


def _indexes(
    reader: PatientCsvReader,
    names: Iterable[str],
) -> list[int]:
    """Find configured columns, rejecting ambiguous duplicates."""
    result = []

    for name in names:
        positions = reader.columns.get(
            _normalise(name),
            [],
        )

        if len(positions) > 1:
            raise ValueError(
                f"Ambiguous duplicate source column: {name}"
            )

        if positions:
            result.append(positions[0])

    return list(dict.fromkeys(result))


def _row_timestamp(
    row: list[str],
    indexes: list[int],
) -> tuple[float, str] | None:
    for index in indexes:
        if index >= len(row):
            continue

        parsed = _timestamp(row[index])

        if parsed is not None:
            return parsed

    return None


def _find_boundaries(
    reader: PatientCsvReader,
    config: ExtractionConfig,
    timestamp_indexes: list[int],
) -> dict[str, float | None]:
    """Find chronological operation boundaries."""
    matches: dict[str, list[float]] = {
        name: []
        for name in config.boundaries
    }

    definitions = {}

    for name, boundary in config.boundaries.items():
        definitions[name] = (
            _indexes(reader, boundary.fields),
            tuple(
                _normalise(phrase)
                for phrase in boundary.phrases
            ),
        )

    for row in reader.rows():
        stamp = _row_timestamp(
            row,
            timestamp_indexes,
        )

        if stamp is None:
            continue

        timestamp = stamp[0]

        for name, (indexes, phrases) in definitions.items():
            for index in indexes:
                if index >= len(row):
                    continue

                value = _normalise(row[index])

                if any(
                    phrase in value
                    for phrase in phrases
                ):
                    matches[name].append(timestamp)
                    break

    starts = matches["operation_start"]
    ends = matches["operation_end"]
    post_ends = matches["postoperative_end"]

    start = min(starts) if starts else None

    end = min(
        (
            value
            for value in ends
            if start is not None and value > start
        ),
        default=None,
    )

    post_end = min(
        (
            value
            for value in post_ends
            if end is not None and value > end
        ),
        default=None,
    )

    return {
        "operation_start": start,
        "operation_end": end,
        "postoperative_end": post_end,
    }


def _phase_at(
    timestamp: float,
    boundaries: dict[str, float | None],
) -> str | None:
    """Return the unique clinical phase of an observation.

    Intervals:
        preoperative:    t < operation_start
        intraoperative:  start <= t < end
        postoperative:   end <= t < postoperative_end

    An absent postoperative_end means no configured upper
    time bound. Missing required boundaries produce no phase.
    """
    start = boundaries["operation_start"]
    end = boundaries["operation_end"]
    post_end = boundaries["postoperative_end"]

    if start is None:
        return None

    if timestamp < start:
        return "preoperative"

    if end is None:
        return None

    if start <= timestamp < end:
        return "intraoperative"

    if timestamp >= end:
        if post_end is None or timestamp < post_end:
            return "postoperative"

    return None


def _safe_patient_id(value: str) -> str:
    """Validate patient ID for safe use as a filename."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(
            f"Unsafe patient ID in filename: {value}"
        )

    if value in {"", ".", ".."}:
        raise ValueError(
            f"Invalid patient ID: {value}"
        )

    return value


def _patient_id(paths: tuple[Path, ...]) -> str:
    """Identify a family using the shared filename utility."""
    siblings = {path.stem for path in paths}

    families = {
        patient_family(path, siblings)[0]
        for path in paths
    }

    if len(families) != 1:
        raise ValueError(
            f"Inconsistent patient filename family: {paths}"
        )

    return _safe_patient_id(next(iter(families)))


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=OUTPUT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_table(
    path: Path,
    fields: list[str],
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_patient(
    paths: tuple[Path, ...],
    config: ExtractionConfig,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, float | None],
    list[dict[str, str]],
]:
    """Extract measurements into the configured phases.

    A timed observation is assigned to exactly one phase.
    It is exported only if that phase is enabled for the
    parameter.
    """
    reader = PatientCsvReader(paths)

    timestamp_indexes = _indexes(
        reader,
        config.timestamp_fields,
    )

    boundaries = _find_boundaries(
        reader,
        config,
        timestamp_indexes,
    )

    selected: dict[str, list[int]] = {}
    manifest: list[dict[str, str]] = []

    for parameter in config.parameters:
        columns = _indexes(
            reader,
            parameter.source_fields,
        )

        selected[parameter.id] = columns

        if config.strict_columns and not columns and parameter.derive == "none":
            raise ValueError(
                f"Missing source columns for {parameter.id}"
            )

        for phase in parameter.phases:
            manifest.append({
                "parameter_id": parameter.id,
                "phase": phase,
                "source_fields": ",".join(
                    parameter.source_fields
                ),
                "matched_columns": str(len(columns)),
                "status": (
                    "matched"
                    if columns
                    else ("derivation_candidate" if parameter.derive != "none" else "column_missing")
                ),
            })

    extracted: dict[
        str,
        list[dict[str, str]],
    ] = {
        phase: []
        for phase in PHASES
    }

    for row in reader.rows():
        stamp = _row_timestamp(
            row,
            timestamp_indexes,
        )

        actual_phase = (
            _phase_at(stamp[0], boundaries)
            if stamp is not None
            else None
        )

        for parameter in config.parameters:
            if parameter.timed:
                if stamp is None or actual_phase is None:
                    continue

                if actual_phase not in parameter.phases:
                    continue

                target_phase = actual_phase
                timestamp_text = format((stamp[0] - boundaries["operation_start"]) / 60.0, ".12g")

            else:
                # An untimed *parameter* may occur in timestamped CSV rows.
                # Do not use a known post-start observation as a pre-op
                # feature. Rows without a timestamp remain eligible for
                # explicit static demographics (and derivation).
                if (
                    stamp is not None
                    and boundaries["operation_start"] is not None
                    and stamp[0] >= boundaries["operation_start"]
                ):
                    continue
                target_phase = "preoperative"
                timestamp_text = ""

            for index in selected[parameter.id]:
                if index >= len(row):
                    continue

                raw = row[index].strip()

                if _normalise(raw) in config.missing_tokens:
                    continue

                value = _parse_value(
                    raw,
                    parameter.type,
                )

                if value is None:
                    continue

                extracted[target_phase].append({
                    "time_minutes": timestamp_text,
                    "parameter_id": parameter.id,
                    "value": value,
                    "unit": parameter.unit,
                    "type": parameter.type,
                    "source_field": reader.headers[index],
                    "value_origin": "observed",
                })

    # Preserve raw Epic event strings as patient observations, with elapsed minutes.
    event_indexes = _indexes(reader, ("epic.EVENTS.EVENT", "epic.EVENT.name"))
    for row in reader.rows():
        stamp = _row_timestamp(row, timestamp_indexes)
        if stamp is None or boundaries["operation_start"] is None:
            continue
        phase = _phase_at(stamp[0], boundaries)
        if phase is None:
            continue
        for index in event_indexes:
            raw = row[index].strip()
            if _normalise(raw) in config.missing_tokens:
                continue
            extracted[phase].append({
                "time_minutes": format((stamp[0] - boundaries["operation_start"]) / 60.0, ".12g"),
                "parameter_id": "epic_event",
                "value": raw,
                "unit": "",
                "type": "str",
                "source_field": reader.headers[index],
                "value_origin": "observed",
            })

    # Derived static features are calculated only from the current patient's
    # preoperative (or explicitly untimed) measurements. No cohort statistics,
    # no postoperative data, and no inferred values without both inputs.
    from src.preprocessing.parameters.derivation import derive_patient_parameters
    derive_patient_parameters(extracted, config, boundaries)

    for phase in PHASES:
        extracted[phase].sort(
            key=lambda item: (
                float(item["time_minutes"]) if item["time_minutes"] else float("-inf"),
                item["parameter_id"],
                item["source_field"],
            )
        )

    return extracted, boundaries, manifest


def extract_parameters(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    config_path: str | Path,
) -> dict[str, int]:
    """Extract phase-specific measurements for each patient."""
    config = load_config(config_path)
    output = Path(output_dir).resolve()

    files = discover_csv_files(
        inputs,
        output,
    )

    if not files:
        raise ValueError(
            "No input CSV files found"
        )

    groups = group_patient_files(files)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for phase in PHASES:
        (output / phase).mkdir(
            parents=True,
            exist_ok=True,
        )

    summary_rows = []
    manifest_rows = []
    used_ids = set()
    total_readings = 0
    total_derived = 0
    event_summary_rows = []

    for group in groups:
        paths = tuple(
            Path(path).resolve()
            for path in group
        )

        patient_id = _patient_id(paths)

        if patient_id in used_ids:
            raise ValueError(
                "Duplicate patient ID across directories: "
                f"{patient_id}"
            )

        used_ids.add(patient_id)

        extracted, boundaries, manifest = extract_patient(
            paths,
            config,
        )

        for item in manifest:
            manifest_rows.append({
                "patient_id": patient_id,
                **item,
            })

        counts = {}

        for phase in PHASES:
            rows = extracted[phase]
            counts[phase] = len(rows)
            total_readings += len(rows)
            total_derived += sum(row.get("value_origin", "") != "observed" for row in rows)

            _write_csv(
                output / phase / f"{patient_id}.csv",
                rows,
            )

        # Count every event string, even if operation boundaries are missing.
        event_reader = PatientCsvReader(paths)
        event_columns = _indexes(event_reader, ("epic.EVENTS.EVENT", "epic.EVENT.name"))
        event_counts = defaultdict(int)
        for raw_row in event_reader.rows():
            for index in event_columns:
                raw_value = raw_row[index].strip()
                if _normalise(raw_value) not in config.missing_tokens:
                    event_counts[(event_reader.headers[index], raw_value)] += 1
        for (source_field, event_value), count in sorted(event_counts.items()):
            event_summary_rows.append({"patient_id": patient_id,
                                       "source_field": source_field, "event_value": event_value,
                                       "occurrences": count})

        summary_rows.append({
            "patient_id": patient_id,
            "source_files": len(paths),
            "operation_start_minutes": (
                0 if boundaries["operation_start"] is not None else ""
            ),
            "operation_end_minutes": (
                format((boundaries["operation_end"] - boundaries["operation_start"]) / 60.0, ".12g")
                if boundaries["operation_end"] is not None and boundaries["operation_start"] is not None else ""
            ),
            "postoperative_end_minutes": (
                format((boundaries["postoperative_end"] - boundaries["operation_start"]) / 60.0, ".12g")
                if boundaries["postoperative_end"] is not None and boundaries["operation_start"] is not None else ""
            ),
            "preoperative_rows": counts["preoperative"],
            "intraoperative_rows": counts["intraoperative"],
            "postoperative_rows": counts["postoperative"],
        })

    _write_table(output / "patient_events_summary.csv",
                 ["patient_id", "source_field", "event_value", "occurrences"],
                 event_summary_rows)

    _write_table(
        output / "summary.csv",
        [
            "patient_id",
            "source_files",
            "operation_start_minutes",
            "operation_end_minutes",
            "postoperative_end_minutes",
            "preoperative_rows",
            "intraoperative_rows",
            "postoperative_rows",
        ],
        summary_rows,
    )

    _write_table(
        output / "manifest.csv",
        [
            "patient_id",
            "parameter_id",
            "phase",
            "source_fields",
            "matched_columns",
            "status",
        ],
        manifest_rows,
    )

    return {
        "files": len(files),
        "patients": len(groups),
        "parameters": len(config.parameters),
        "parameter_phase_assignments": sum(
            len(parameter.phases)
            for parameter in config.parameters
        ),
        "readings": total_readings,
        "derived_readings": total_derived,
    }
