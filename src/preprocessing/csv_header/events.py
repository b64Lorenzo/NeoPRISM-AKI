"""Streaming inventory of Epic event strings; no patient identifiers in global report."""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
from pathlib import Path
from src.utils.csv_utils import csv_encoding_and_dialect, normalise
from src.utils.patient_files import discover_csv_files, group_patient_files, patient_family

EVENT_FIELDS = ('epic.EVENTS.EVENT', 'epic.EVENT.name')
MISSING = {'', 'na', 'n/a', 'null', 'none', 'nan'}

def _md(value: object) -> str:
    return str(value).replace('\\', '\\\\').replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')

def write_event_reports(inputs, output_directory):
    """Generate global event values and patient-by-event counts."""
    out = Path(output_directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = discover_csv_files(inputs, out)
    groups = group_patient_files(files)
    global_counts = Counter()
    global_files = defaultdict(set)
    per_patient = Counter()
    matched_columns = Counter()
    missing_cells = 0
    patients_with_events = 0
    used_ids = set()
    for group in groups:
        siblings = {p.stem for p in group}
        patient_ids = {patient_family(p, siblings)[0] for p in group}
        if len(patient_ids) != 1:
            raise ValueError(f'Inconsistent patient family: {group}')
        patient_id = next(iter(patient_ids))
        if patient_id in used_ids:
            raise ValueError(f'Duplicate patient ID across directories: {patient_id}')
        used_ids.add(patient_id)
        any_event = False
        for path in group:
            encoding, dialect = csv_encoding_and_dialect(path)
            with path.open(encoding=encoding, newline='') as stream:
                reader = csv.DictReader(stream, dialect=dialect)
                fields = [f for f in (reader.fieldnames or []) if f and normalise(f) in {normalise(x) for x in EVENT_FIELDS}]
                for field in set(fields):
                    matched_columns[field] += 1
                for row in reader:
                    for field in fields:
                        value = (row.get(field) or '').strip()
                        if normalise(value) in MISSING:
                            missing_cells += 1
                            continue
                        any_event = True
                        key = (field, value)
                        global_counts[key] += 1
                        global_files[key].add(path)
                        per_patient[(patient_id, field, value)] += 1
        patients_with_events += int(any_event)
    global_rows = sorted(global_counts, key=lambda k: (k[0].casefold(), -global_counts[k], k[1].casefold()))
    with (out / 'event_values.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_field', 'event_value', 'occurrences', 'csv_files'])
        for field, value in global_rows:
            key = (field, value)
            writer.writerow([field, value, global_counts[key], len(global_files[key])])
    with (out / 'patient_events.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['patient_id', 'source_field', 'event_value', 'occurrences'])
        for key in sorted(per_patient, key=lambda k: (k[0], k[1], k[2])):
            writer.writerow([*key, per_patient[key]])
    lines = ['# Epic event values summary', '', 'All distinct nonmissing event strings. This global report contains no patient identifiers.', '',
             f'- Patient families scanned: {len(groups)}', f'- Patients with event values: {patients_with_events}',
             f'- Distinct field/value pairs: {len(global_rows)}', f'- Event observations: {sum(global_counts.values())}',
             f'- Missing event cells: {missing_cells}', '', '## Detected source fields', '', '| Field | CSV files |', '|---|---:|']
    for field, count in sorted(matched_columns.items()):
        lines.append(f'| {_md(field)} | {count} |')
    if not matched_columns:
        lines.append('| No matching event column found | 0 |')
    lines += ['', '## All event values', '', '| Source field | Exact string value | Occurrences | CSV files |', '|---|---|---:|---:|']
    for key in global_rows:
        lines.append(f'| {_md(key[0])} | {_md(key[1])} | {global_counts[key]} | {len(global_files[key])} |')
    (out / 'event_values_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'event_values': len(global_rows), 'event_observations': sum(global_counts.values()), 'patients_with_events': patients_with_events}
