"""Configuration for patient-level modelling inputs."""
from dataclasses import dataclass
from pathlib import Path
import configparser

PHASES = ('preoperative','intraoperative','postoperative')
AGGREGATIONS = ('patient_mean','patient_median','patient_min','patient_max','patient_first','patient_last')

@dataclass(frozen=True)
class InputConfig:
    extracted_dir: Path
    parameters_ini: Path
    labels_csv: Path
    output_dir: Path
    features: tuple[str,...]
    aggregation: str
    label_column: str
    label_kind: str
    cutoff_minutes: float | None
    label_settings: object = None


def load_config(path: str | Path) -> InputConfig:
    path=Path(path).resolve()
    c=configparser.ConfigParser(interpolation=None)
    if not c.read(path,encoding='utf8'):
        raise FileNotFoundError(path)
    root=path.parent.parent
    def resolved(section,key,default):
        v=Path(c.get(section,key,fallback=default))
        return v if v.is_absolute() else (root/v).resolve()
    raw=c.get('features','parameter_ids',fallback='').strip()
    features=tuple(v.strip() for v in raw.split(',') if v.strip())
    if not features or len(set(features))!=len(features):
        raise ValueError('features.parameter_ids requires distinct phase/parameter_id names')
    for f in features:
        phase,sep,parameter=f.partition('/')
        if not sep or phase not in PHASES or not parameter:
            raise ValueError(f'Feature must be phase/parameter_id: {f}')
    aggregation=c.get('features','aggregation',fallback='patient_mean')
    if aggregation not in AGGREGATIONS:
        raise ValueError(f'Unsupported aggregation: {aggregation}')
    kind=c.get('labels','kind',fallback='binary').lower()
    if kind not in ('binary','kdigo_stage'):
        raise ValueError('labels.kind must be binary or kdigo_stage')
    raw_cutoff=c.get('features','cutoff_minutes',fallback='').strip()
    cutoff=float(raw_cutoff) if raw_cutoff else None
    if cutoff is not None and (cutoff < 0 or not __import__('math').isfinite(cutoff)):
        raise ValueError('cutoff_minutes must be finite and nonnegative')
    if cutoff is not None and any(f.startswith('postoperative/') for f in features):
        raise ValueError('Postoperative features cannot be used for intraoperative cutoff prediction')
    from src.preprocessing.inputsModel.clinical_labels import ClinicalLabelConfig
    from src.preprocessing.inputsModel.label_strategy import LabelSettings
    label_settings=LabelSettings(
        source=c.get('labels','source',fallback='external').strip().lower(),
        csv_path=resolved('paths','labels_csv','data/labels/aki_labels.csv'),
        column=c.get('labels','column',fallback='aki_label'),
        kind=kind,
        clinical=ClinicalLabelConfig(
            creatinine_parameter_id=c.get('clinical_aki','creatinine_parameter_id',fallback='kreatinine'),
            creatinine_unit=c.get('clinical_aki','creatinine_unit',fallback='umol/L'),
            baseline_strategy=c.get('clinical_aki','baseline_strategy',fallback='preoperative_last'),
            window_hours=c.getfloat('clinical_aki','window_hours',fallback=72),
            threshold_profile=c.get('clinical_aki','threshold_profile',fallback='neonatal_2017'),
            include_intraoperative_baseline=c.getboolean('clinical_aki','include_intraoperative_baseline',fallback=False),
        ),
    )
    return InputConfig(resolved('paths','input_directory','results/parameters'),
        resolved('paths','parameter_config','config/parameters.ini'),
        resolved('paths','labels_csv','data/labels/aki_labels.csv'),
        resolved('paths','output_directory','results/inputsModel'),
        features,aggregation,c.get('labels','column',fallback='aki_label'),kind,cutoff,label_settings)
