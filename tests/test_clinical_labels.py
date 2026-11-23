import csv
from src.preprocessing.inputsModel.clinical_labels import ClinicalLabelConfig, derive_labels
from src.preprocessing.inputsModel.label_strategy import LabelSettings, resolve_labels
from src.models.rosenblatt_perceptron.training import read_matrix


def create_patient(root, pid, pre, post, end=120):
    for phase,values in [('preoperative',pre),('postoperative',post)]:
        folder=root/phase;folder.mkdir(parents=True,exist_ok=True)
        with (folder/f'{pid}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['time_minutes','parameter_id','value','unit','type','source_field'])
            for t,v in values:w.writerow([t,'kreatinine',v,'umol/L','numeric','epic.creatinine'])


def test_creatinine_stage_and_missing(tmp_path):
    create_patient(tmp_path,'ref_1',[(-60,88.4)],[(180,176.8)])
    create_patient(tmp_path,'ref_2',[(-60,88.4)],[(180,88.4)])
    create_patient(tmp_path,'ref_3',[(-60,88.4)],[])
    (tmp_path/'summary.csv').write_text('patient_id,operation_end_minutes\nref_1,120\nref_2,120\nref_3,120\n')
    r=derive_labels(tmp_path,['ref_1','ref_2','ref_3'],ClinicalLabelConfig())
    assert r['ref_1']['stage']==2
    assert r['ref_2']['stage']==0
    assert r['ref_3']['stage'] is None
    labels=resolve_labels(tmp_path,['ref_1','ref_2','ref_3'],LabelSettings(source='creatinine'))
    assert [labels[k]['binary_label'] for k in ['ref_1','ref_2','ref_3']]==[1,-1,None]


def test_neonatal_stage3_threshold(tmp_path):
    create_patient(tmp_path,'ref_1',[(-60,176.8)],[(180,221.0)])
    (tmp_path/'summary.csv').write_text('patient_id,operation_end_minutes\nref_1,120\n')
    assert derive_labels(tmp_path,['ref_1'],ClinicalLabelConfig())['ref_1']['stage']==3
    assert derive_labels(tmp_path,['ref_1'],ClinicalLabelConfig(threshold_profile='kdigo_2012'))['ref_1']['stage']==0


def test_model_does_not_consume_label_metadata(tmp_path):
    f=tmp_path/'matrix.csv'
    f.write_text('patient_id,preoperative/age,aki_label,aki_stage,label_status,label_reason\np1,4,1,2,determined,creatinine_only\n')
    _,features=read_matrix(f)
    assert features==['preoperative/age']
