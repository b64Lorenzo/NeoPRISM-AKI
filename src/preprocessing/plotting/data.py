"""Read per-patient, per-phase extraction CSV files."""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from src.preprocessing.plotting.selection import ParameterInfo

REQUIRED_COLUMNS = {'time_minutes', 'parameter_id', 'value', 'unit', 'type', 'source_field'}
PHASES = ('preoperative', 'intraoperative', 'postoperative')

@dataclass(frozen=True)
class Observation:
    patient_id: str
    phase: str
    parameter_id: str
    timestamp: float | None
    value: str
    numeric_value: float | None
    source_field: str = ""
    value_origin: str = "observed"

    @property
    def key(self) -> str:
        return f'{self.phase}/{self.parameter_id}'


def load_observations(input_dir: Path, catalog: dict[str, ParameterInfo], selected_patients: tuple[str, ...] = ()) -> tuple[list[Observation], list[str]]:
    observations = []
    patients = set()
    wanted = set(selected_patients)
    for phase in PHASES:
        if not any(p.phase == phase for p in catalog.values()):
            continue
        directory = input_dir / phase
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob('*.csv')):
            patient = path.stem
            if wanted and patient not in wanted:
                continue
            patients.add(patient)
            with path.open('r', encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
                    raise ValueError(f'Invalid extraction CSV: {path}')
                for row in reader:
                    identifier = (row['parameter_id'] or '').strip()
                    parameter = catalog.get(f'{phase}/{identifier}')
                    if parameter is None:
                        continue
                    raw_time = (row['time_minutes'] or '').strip()
                    try:
                        timestamp = float(raw_time) if raw_time else None
                        if timestamp is not None and not math.isfinite(timestamp):
                            raise ValueError('Non-finite elapsed time')
                    except ValueError as exc:
                        raise ValueError(f'Invalid time_minutes in {path}: {raw_time}') from exc
                    value = (row['value'] or '').strip()
                    numeric = None
                    if parameter.type in {'numeric', 'timeseries'}:
                        try:
                            candidate = float(value)
                            if math.isfinite(candidate):
                                numeric = candidate
                        except ValueError:
                            pass
                    observations.append(Observation(
                        patient, phase, identifier, timestamp, value, numeric,
                        (row.get("source_field") or "").strip(),
                        (row.get("value_origin") or "observed").strip(),
                    ))
    unknown = wanted - patients
    if unknown:
        raise ValueError(f'Unknown selected patients: {sorted(unknown)}')
    return observations, sorted(patients)


def by_parameter(observations: list[Observation]) -> dict[str, list[Observation]]:
    result = defaultdict(list)
    for item in observations:
        result[item.key].append(item)
    return dict(result)


def patient_numeric_values(observations: list[Observation], aggregation: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in observations:
        if row.numeric_value is not None:
            grouped[row.patient_id].append(row.numeric_value)
    operations = {'patient_mean': mean, 'patient_median': median, 'patient_min': min, 'patient_max': max}
    if aggregation not in operations:
        raise ValueError(f'Unknown aggregation: {aggregation}')
    return {patient: operations[aggregation](values) for patient, values in grouped.items()}


def load_patient_boundaries(input_dir: Path, patients: list[str]) -> dict[str, dict[str, float | None]]:
    """Load extractor-computed operation-relative surgical boundaries.

    Do not guess operation end from the last intraoperative measurement: only
    the extraction manifest summary is authoritative for the selected boundary.
    """
    result = {patient: {"operation_start": None, "operation_end": None,
                        "postoperative_end": None} for patient in patients}
    summary = input_dir / "summary.csv"
    if not summary.is_file():
        return result
    mapping = {"operation_start_minutes": "operation_start",
               "operation_end_minutes": "operation_end",
               "postoperative_end_minutes": "postoperative_end"}
    with summary.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            pid = (row.get("patient_id") or "").strip()
            if pid not in result:
                continue
            for column, key in mapping.items():
                raw = (row.get(column) or "").strip()
                if not raw:
                    continue
                try:
                    number = float(raw)
                except ValueError as error:
                    raise ValueError(f"Invalid {column} for patient {pid}: {raw}") from error
                if not math.isfinite(number):
                    raise ValueError(f"Non-finite {column} for patient {pid}")
                result[pid][key] = number
    return result


def load_patient_events(input_dir: Path, patients: list[str]) -> dict[str, list[Observation]]:
    """Read extracted Epic events without depending on the selected numeric catalog.

    The extractor automatically exports `epic_event` observations even if they
    are not declared as [parameter.events] in parameters.ini. Prefer those
    canonical events, avoiding duplicates from manually configured event fields.
    """
    result: dict[str, list[Observation]] = {patient: [] for patient in patients}
    requested = set(patients)
    for phase in PHASES:
        folder = input_dir / phase
        if not folder.is_dir():
            continue
        for file in sorted(folder.glob("*.csv")):
            if file.stem not in requested:
                continue
            with file.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"parameter_id", "value", "time_minutes"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Invalid extraction CSV: {file}")
                for row in reader:
                    if (row.get("parameter_id") or "").strip() != "epic_event":
                        continue
                    value = (row.get("value") or "").strip()
                    if not value:
                        continue
                    raw = (row.get("time_minutes") or "").strip()
                    if not raw:
                        continue
                    try:
                        minutes = float(raw)
                        if not math.isfinite(minutes):
                            raise ValueError("non-finite")
                    except ValueError as exc:
                        raise ValueError(f"Invalid event time_minutes in {file}: {raw}") from exc
                    result[file.stem].append(Observation(
                        file.stem, phase, "epic_event", minutes, value, None,
                        (row.get("source_field") or "epic.EVENTS.EVENT").strip(),
                        (row.get("value_origin") or "observed").strip(),
                    ))
    for events in result.values():
        events.sort(key=lambda event: (event.timestamp, event.value))
    return result
