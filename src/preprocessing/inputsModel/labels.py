"""Strict external label ingestion; never equate unknown with No AKI."""
import csv
from pathlib import Path

UNKNOWN={'','unknown','undetermined','na','n/a','none','null','?'}

def parse_label(raw: str, kind: str) -> int | None:
    value=raw.strip().casefold()
    if value in UNKNOWN:
        return None
    if kind=='binary':
        mapping={'-1':-1,'+1':1,'1':1,'no aki':-1,'yes aki':1,'no':-1,'yes':1}
        if value not in mapping:
            raise ValueError(f'Invalid binary AKI label: {raw!r}')
        return mapping[value]
    if kind=='kdigo_stage':
        if value not in {'0','1','2','3'}:
            raise ValueError(f'Invalid KDIGO stage: {raw!r}')
        return -1 if value=='0' else 1
    raise ValueError(f'Unknown label kind: {kind}')

def read_labels(path: Path, column: str, kind: str) -> dict[str,int|None]:
    if not path.exists():
        raise FileNotFoundError(f'Label CSV not found: {path}. Adjudicate labels before training.')
    result={}
    with path.open(newline='',encoding='utf-8-sig') as file:
        reader=csv.DictReader(file)
        if not {'patient_id',column}.issubset(reader.fieldnames or ()):
            raise ValueError(f'Label CSV requires patient_id and {column}')
        for row in reader:
            pid=(row['patient_id'] or '').strip()
            if not pid or pid in result:
                raise ValueError(f'Empty/duplicate patient_id in label CSV: {pid!r}')
            result[pid]=parse_label(row[column] or '',kind)
    return result
