import csv
import json
from src.models.rosenblatt_perceptron.core import train
from src.models.rosenblatt_perceptron.training import fit
from src.models.rosenblatt_perceptron.execution.predict import predict_csv
from src.models.rosenblatt_perceptron.validation.metrics import evaluate,auc

def test_linearly_separable():
    model,info=train([[2,1],[-2,-1],[3,1],[-3,-1]],[1,-1,1,-1],epochs=100)
    assert info['converged']
    assert model.predict([2,1])==1 and model.predict([-2,-1])==-1

def test_metrics():
    m=evaluate([-1,-1,1,1],[-1,1,-1,1],[-2,1,-1,2])
    assert m['confusion_matrix']==[[1,1],[1,1]]
    assert m['accuracy']==.5
    assert auc([-1,1],[-1,1])==1

def test_pipeline_and_novel_inference(tmp_path):
    matrix=tmp_path/'matrix.csv'
    with matrix.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['patient_id','preoperative/age','preoperative/sex','aki_label'])
        for i in range(12):w.writerow([f'case_{i}',i,'F' if i%2 else 'M',1 if i>=6 else -1])
        w.writerow(['unlabeled',2,'F',''])
    out=tmp_path/'trained'
    result=fit(matrix,out,seed=3,epochs=30)
    assert result['training']['excluded_unlabeled']==1
    ids=json.loads((out/'split.json').read_text())
    assert not set(ids['training_patient_ids'])&set(ids['validation_patient_ids'])
    assert (out/'validation'/'metrics.json').exists()
    novel=tmp_path/'novel.csv'
    with novel.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['patient_id','preoperative/age','preoperative/sex'])
        w.writerow(['new_patient',5,'M'])
    assert predict_csv(out/'model.json',novel,tmp_path/'predictions.csv')==1
