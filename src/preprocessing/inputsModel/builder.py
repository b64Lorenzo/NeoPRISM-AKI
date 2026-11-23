"""One row per patient/episode; no model fitting in this stage."""
import csv
import json
from collections import defaultdict
from statistics import mean,median
from src.preprocessing.inputsModel.config import InputConfig
from src.preprocessing.inputsModel.label_strategy import resolve_labels
from src.preprocessing.plotting.data import load_observations
from src.preprocessing.plotting.selection import ParameterInfo
from src.preprocessing.parameters.config import load_config as load_parameter_config


def _aggregate(rows, kind, how):
    if not rows:
        return ''
    if kind in ('numeric','timeseries'):
        values=[r.numeric_value for r in rows if r.numeric_value is not None]
        if not values: return ''
        if how=='patient_mean': return format(mean(values),'.12g')
        if how=='patient_median': return format(median(values),'.12g')
        if how=='patient_min': return format(min(values),'.12g')
        if how=='patient_max': return format(max(values),'.12g')
        if how=='patient_first': return format(values[0],'.12g')
        if how=='patient_last': return format(values[-1],'.12g')
    # Categorical features: select first/last nonempty observation, never average.
    values=[r.value for r in rows if r.value.strip()]
    return (values[-1] if how=='patient_last' else values[0]) if values else ''


def build(config: InputConfig, *, require_labels: bool=True) -> dict:
    params=load_parameter_config(config.parameters_ini)
    catalog={}
    for par in params.parameters:
        phases=getattr(par,'phases',None) or tuple(x.strip() for x in par.phase.split(','))
        for phase in phases:
            key=f'{phase}/{par.id}'
            if key in config.features:
                catalog[key]=ParameterInfo(par.id,phase,par.type,par.unit,tuple(par.source_fields),(),())
    unknown=set(config.features)-set(catalog)
    if unknown:
        raise ValueError(f'Unknown feature parameter-phase IDs: {sorted(unknown)}')
    observations,patients=load_observations(config.extracted_dir,catalog)
    labels=resolve_labels(config.extracted_dir,patients,config.label_settings) if require_labels else {}
    grouped=defaultdict(list)
    for row in observations:
        if config.cutoff_minutes is not None and row.phase=='intraoperative':
            if row.timestamp is None or row.timestamp>config.cutoff_minutes:
                continue
        grouped[(row.patient_id,row.key)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r:(r.timestamp is None, r.timestamp if r.timestamp is not None else 0))
    config.output_dir.mkdir(parents=True,exist_ok=True)
    output=config.output_dir/('patients_labeled.csv' if require_labels else 'patients_unlabeled.csv')
    fields=['patient_id',*config.features]+(['aki_label','aki_stage','label_status','label_reason'] if require_labels else [])
    excluded=0; counts={-1:0,1:0}; missing=defaultdict(int)
    with output.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for pid in patients:
            row={'patient_id':pid}
            for feature in config.features:
                row[feature]=_aggregate(grouped[(pid,feature)],catalog[feature].type,config.aggregation)
                if not row[feature]: missing[feature]+=1
            if require_labels:
                result=labels.get(pid,{})
                label=result.get('binary_label')
                row['aki_label']='' if label is None else label
                row['aki_stage']='' if result.get('stage') is None else result['stage']
                row['label_status']=result.get('status','unknown')
                row['label_reason']=result.get('reason','no_label')
                if label is None: excluded+=1
                else: counts[label]+=1
            writer.writerow(row)
    report={'input_file':str(output),'patients':len(patients),'features':list(config.features),
        'missing_by_feature':dict(missing),'unlabeled_patients':excluded if require_labels else None,
        'labeled_no_aki':counts[-1] if require_labels else None,
        'labeled_aki':counts[1] if require_labels else None,
        'cutoff_minutes':config.cutoff_minutes,'label_kind':config.label_kind if require_labels else None, 'label_source':config.label_settings.source if require_labels else None}
    (config.output_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report
