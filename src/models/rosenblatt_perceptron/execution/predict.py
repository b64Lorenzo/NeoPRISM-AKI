import csv
import json
from pathlib import Path
from src.models.rosenblatt_perceptron.core import Perceptron
from src.models.rosenblatt_perceptron.transform import Transformer

def load_artifact(path):
    data=json.loads(Path(path).read_text(encoding='utf8'))
    if data.get('schema_version')!=1 or data.get('model_type')!='binary_rosenblatt_perceptron':
        raise ValueError('Unsupported model artifact')
    transform=Transformer.from_dict(data['transformer'])
    weights=data['weights'];bias=data['bias']
    expected=len(transform.transform_one({}))
    if not isinstance(weights,list) or len(weights)!=expected:
        raise ValueError('Model weights/transform dimension mismatch')
    model=Perceptron([float(w) for w in weights],float(bias),float(data['threshold']))
    return model,transform

def predict_csv(artifact,input_csv,output_csv):
    model,transform=load_artifact(artifact)
    with Path(input_csv).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'patient_id',*transform.features}.issubset(reader.fieldnames or ()):
            raise ValueError('Novel data schema does not match saved model')
        rows=list(reader)
    if len({r['patient_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate patient IDs in novel data')
    output=Path(output_csv);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['patient_id','predicted_label','decision_score'])
        writer.writeheader()
        for row in rows:
            score=model.score(transform.transform_one(row))
            writer.writerow({'patient_id':row['patient_id'],'predicted_label':1 if score>=0 else -1,'decision_score':score})
    return len(rows)
