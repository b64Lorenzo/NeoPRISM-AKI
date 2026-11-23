"""Research-only configurable AKI outcome derivation from extracted patient data.

Only serum-creatinine staging is automated here. Urine-output staging requires
validated interval coverage and is deliberately not inferred from spot readings.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import math

from src.preprocessing.plotting.data import load_observations
from src.preprocessing.plotting.selection import ParameterInfo


@dataclass(frozen=True)
class ClinicalLabelConfig:
    creatinine_parameter_id: str = 'kreatinine'
    creatinine_unit: str = 'umol/L'
    baseline_strategy: str = 'preoperative_last'  # preoperative_last, preoperative_min, previous_lowest
    window_hours: float = 72.0
    threshold_profile: str = 'neonatal_2017'  # neonatal_2017, kdigo_2012
    include_intraoperative_baseline: bool = False
    require_baseline: bool = True


def _mg_dl(value: float, unit: str) -> float:
    normalized = unit.lower().replace('µ','u').replace('μ','u').replace(' ','')
    if normalized in ('umol/l','umol/liter','umol/litre'):
        return value / 88.4
    if normalized in ('mg/dl','mg/dl.'):
        return value
    raise ValueError(f'Unsupported creatinine unit: {unit!r}; choose umol/L or mg/dL')


def derive_labels(extracted_dir, patient_ids: list[str], options: ClinicalLabelConfig):
    """Return {patient_id: {'stage': int|None, 'status': str, 'reason': str}}.

    Postoperative time is minutes relative to surgery start, NOT relative to
    surgery end. The latter is read from extraction summary.csv. A stage 0 is
    assigned only when a baseline and >=1 postoperative measurement exist;
    it means *no creatinine-defined AKI observed*, not verified absence of AKI.
    """
    if options.baseline_strategy not in {'preoperative_last','preoperative_min','previous_lowest'}:
        raise ValueError('Invalid baseline_strategy')
    if options.threshold_profile not in {'neonatal_2017','kdigo_2012'}:
        raise ValueError('Invalid threshold_profile')
    if not math.isfinite(options.window_hours) or options.window_hours <= 0:
        raise ValueError('window_hours must be positive and finite')
    catalog={f'{phase}/{options.creatinine_parameter_id}': ParameterInfo(
        options.creatinine_parameter_id,phase,'numeric',options.creatinine_unit,(),(),())
        for phase in ('preoperative','intraoperative','postoperative')}
    rows,_=load_observations(extracted_dir,catalog)
    grouped=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row.numeric_value is not None and row.numeric_value > 0:
            grouped[row.patient_id][row.phase].append((row.timestamp,_mg_dl(row.numeric_value,options.creatinine_unit)))
    from src.preprocessing.plotting.data import load_patient_boundaries
    boundaries=load_patient_boundaries(extracted_dir,patient_ids)
    results={}
    for pid in patient_ids:
        pre=grouped[pid]['preoperative'][:]
        if options.include_intraoperative_baseline:
            pre+=grouped[pid]['intraoperative']
        pre=[(t,v) for t,v in pre if t is not None]
        pre.sort()
        end=boundaries[pid]['operation_end']
        if end is None:
            results[pid]={'stage':None,'status':'unknown','reason':'operation_end_missing'}
            continue
        post=sorted((t,v) for t,v in grouped[pid]['postoperative']
                    if t is not None and end <= t <= end+60*options.window_hours)
        if not post:
            results[pid]={'stage':None,'status':'unknown','reason':'no_postoperative_creatinine_in_window'}
            continue
        if not pre:
            results[pid]={'stage':None,'status':'unknown','reason':'no_valid_preoperative_baseline'}
            continue
        baseline=(min(v for _,v in pre) if options.baseline_strategy in {'preoperative_min','previous_lowest'}
                  else pre[-1][1])
        stage=0
        for t,v in post:
            ref=baseline
            if options.baseline_strategy=='previous_lowest':
                earlier=[val for tm,val in post if tm < t]
                ref=min([baseline,*earlier])
            if v >= 3*ref or v >= (2.5 if options.threshold_profile=='neonatal_2017' else 4.0):
                stage=max(stage,3)
            elif v >= 2*ref:
                stage=max(stage,2)
            elif v >= 1.5*ref:
                stage=max(stage,1)
            # 0.3 mg/dL increase within 48 hours requires a *timed* comparator.
            # Preoperative values lack a guaranteed 48h recency. Use only timed
            # postoperative pairs here to avoid a spurious 48h classification.
            if any(0 < t-previous_t <= 2880 and v-previous_v >= 0.3-1e-9
                   for previous_t,previous_v in post if previous_t<t):
                stage=max(stage,1)
        results[pid]={'stage':stage,'status':'determined','reason':'creatinine_only',
                      'baseline_mg_dl':baseline,'postoperative_samples':len(post)}
    return results
