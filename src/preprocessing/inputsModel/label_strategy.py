"""Pluggable label sources: external, creatinine, external_override, consensus."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
from src.preprocessing.inputsModel.clinical_labels import ClinicalLabelConfig, derive_labels
from src.preprocessing.inputsModel.labels import parse_label

@dataclass(frozen=True)
class LabelSettings:
    source: str = 'external'  # external, creatinine, external_override, consensus
    csv_path: Path | None = None
    column: str = 'aki_label'
    kind: str = 'binary'
    clinical: ClinicalLabelConfig = ClinicalLabelConfig()


def resolve_labels(extracted_dir: Path, patients: list[str], settings: LabelSettings):
    if settings.source not in {'external','creatinine','external_override','consensus'}:
        raise ValueError('labels.source must be external, creatinine, external_override, or consensus')
    external={}
    if settings.source != 'creatinine':
        if settings.csv_path is None or not settings.csv_path.is_file():
            raise FileNotFoundError(f'External label CSV required by source={settings.source}: {settings.csv_path}. '
                                    'Use labels.source=creatinine to derive provisional labels from extracted measurements.')
        with settings.csv_path.open(newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            if not {'patient_id',settings.column}.issubset(reader.fieldnames or []):
                raise ValueError(f'External CSV requires patient_id and {settings.column}')
            for row in reader:
                pid=(row['patient_id'] or '').strip()
                if not pid or pid in external:
                    raise ValueError('Missing or duplicate patient_id in external labels')
                raw=(row[settings.column] or '').strip()
                if settings.kind=='kdigo_stage':
                    if raw.casefold() in {'','unknown','na','n/a'}: stage=None
                    elif raw in {'0','1','2','3'}: stage=int(raw)
                    else: raise ValueError(f'Invalid stage {raw!r} for {pid}')
                else:
                    binary=parse_label(raw,'binary')
                    stage=None if binary is None else (0 if binary==-1 else 1)
                external[pid]={'stage':stage,'status':'determined' if stage is not None else 'unknown',
                               'reason':'external_adjudicated'}
    clinical=derive_labels(extracted_dir,patients,settings.clinical) if settings.source!='external' else {}
    result={}
    for pid in patients:
        ext=external.get(pid,{'stage':None,'status':'unknown','reason':'no_external_label'})
        auto=clinical.get(pid,{'stage':None,'status':'unknown','reason':'no_clinical_label'})
        if settings.source=='external': chosen=ext
        elif settings.source=='creatinine': chosen=auto
        elif settings.source=='external_override': chosen=ext if ext['stage'] is not None else auto
        else:
            if ext['stage'] is not None and auto['stage'] is not None and (ext['stage']>0)==(auto['stage']>0):
                chosen={**ext,'reason':'external_and_creatinine_agree'}
            else:
                chosen={'stage':None,'status':'unknown','reason':'missing_or_discordant_labels'}
        result[pid]={**chosen,'binary_label':None if chosen['stage'] is None else (1 if chosen['stage']>0 else -1)}
    return result
