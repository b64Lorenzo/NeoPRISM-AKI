"""Patient-disjoint stratified holdout, training-only transforms and model export."""
import csv
import json
import random
from pathlib import Path
from src.models.rosenblatt_perceptron.core import train
from src.models.rosenblatt_perceptron.transform import Transformer
from src.models.rosenblatt_perceptron.validation.report import save_report
from src.models.rosenblatt_perceptron.plots.charts import save_training_plots

def read_matrix(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames or []
        if 'patient_id' not in fields or 'aki_label' not in fields:
            raise ValueError('Input matrix requires patient_id and aki_label')
        features=[f for f in fields if f not in ('patient_id','aki_label','aki_stage','label_status','label_reason')]
        if not features:raise ValueError('No model features')
        rows=list(reader)
    if any(not r['patient_id'].strip() for r in rows) or len({r['patient_id'] for r in rows})!=len(rows):
        raise ValueError('Patient IDs must be unique and nonempty')
    return rows,features

def split(rows,fraction,seed):
    if not 0<fraction<1:raise ValueError('train_fraction must be in (0,1)')
    groups={-1:[],1:[]};excluded=[]
    for r in rows:
        raw=(r.get('aki_label') or '').strip()
        if not raw:excluded.append(r);continue
        if raw not in ('-1','1','+1'):raise ValueError(f'Invalid AKI label {raw!r}')
        groups[int(raw)].append(r)
    if not groups[-1] or not groups[1]:raise ValueError('Training requires both No AKI and AKI labels')
    rng=random.Random(seed);train_rows=[];validation=[]
    for label in (-1,1):
        group=groups[label][:];rng.shuffle(group)
        if len(group)<2:raise ValueError(f'Need >=2 patients in class {label} for stratified validation')
        n=max(1,min(len(group)-1,round(len(group)*(1-fraction))))
        validation.extend(group[:n]);train_rows.extend(group[n:])
    rng.shuffle(train_rows);rng.shuffle(validation)
    return train_rows,validation,len(excluded)

def fit(matrix,output_dir,*,train_fraction=.8,seed=2026,epochs=100,learning_rate=None,threshold=0.0):
    rows,features=read_matrix(matrix)
    training,validation,excluded=split(rows,train_fraction,seed)
    transformer=Transformer.fit(training,features)
    X=transformer.transform(training);y=[int(r['aki_label']) for r in training]
    model,stats=train(X,y,epochs=epochs,learning_rate=learning_rate,threshold=threshold,seed=seed)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    history=stats.pop('history')
    save_training_plots(history,output/'plots'/'convergence')
    artifact={'schema_version':1,'model_type':'binary_rosenblatt_perceptron',
        'labels':{'-1':'No AKI','1':'AKI'},'weights':model.weights,'bias':model.bias,
        'threshold':model.threshold,'transformer':transformer.to_dict(),
        'training':{**stats,'train_fraction':train_fraction,'training_patients':len(training),
                    'validation_patients':len(validation),'excluded_unlabeled':excluded}}
    (output/'model.json').write_text(json.dumps(artifact,indent=2,allow_nan=False),encoding='utf8')
    xv=transformer.transform(validation);scores=[model.score(x) for x in xv]
    pred=[1 if s>=0 else -1 for s in scores]
    metrics=save_report(output/'validation',[r['patient_id'] for r in validation],
                        [int(r['aki_label']) for r in validation],pred,scores)
    (output/'split.json').write_text(json.dumps({'training_patient_ids':[r['patient_id'] for r in training],
        'validation_patient_ids':[r['patient_id'] for r in validation],
        'excluded_unlabeled':excluded},indent=2),encoding='utf8')
    return {'artifact':str(output/'model.json'),'metrics':metrics,'training':artifact['training']}
