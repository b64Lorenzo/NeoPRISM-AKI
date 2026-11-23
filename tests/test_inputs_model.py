import csv
from pathlib import Path
import pytest
from src.preprocessing.inputsModel.config import load_config
from src.preprocessing.inputsModel.builder import build
from src.preprocessing.inputsModel.labels import parse_label


def test_binary_labels_and_unknown():
    assert parse_label('No AKI','binary') == -1
    assert parse_label('Yes AKI','binary') == 1
    assert parse_label('','binary') is None
    assert parse_label('0','kdigo_stage') == -1
    assert parse_label('3','kdigo_stage') == 1
    with pytest.raises(ValueError):
        parse_label('0','binary')


def test_inputs_builder_and_unknown_labels(tmp_path):
    root=tmp_path
    (root/'config').mkdir()
    (root/'data/labels').mkdir(parents=True)
    (root/'results/parameters/preoperative').mkdir(parents=True)
    (root/'config/parameters.ini').write_text('''[settings]
missing_tokens = NA
[timestamps]
fields = timestamp
[boundary.operation_start]
fields = event
phrases = start
[boundary.operation_end]
fields = event
phrases = end
[parameter.age]
phase = preoperative
source_fields = epic.age
type = numeric
unit = months
timed = false
[parameter.sex]
phase = preoperative
source_fields = epic.sex
type = str
unit =
timed = false
''')
    (root/'config/inputsModel.ini').write_text('''[paths]
input_directory = results/parameters
parameter_config = config/parameters.ini
labels_csv = data/labels/aki_labels.csv
output_directory = results/inputsModel
[features]
parameter_ids = preoperative/age, preoperative/sex
aggregation = patient_mean
[labels]
kind = binary
column = aki_label
''')
    (root/'data/labels/aki_labels.csv').write_text('patient_id,aki_label\nref_001,-1\nref_002,\n')
    for pid,age,sex in [('ref_001','5','F'),('ref_002','7','M')]:
        with (root/f'results/parameters/preoperative/{pid}.csv').open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['time_minutes','parameter_id','value','unit','type','source_field'])
            w.writerow(['','age',age,'months','numeric','epic.age'])
            w.writerow(['','sex',sex,'','str','epic.sex'])
    config=load_config(root/'config/inputsModel.ini')
    result=build(config)
    assert result['patients']==2
    assert result['labeled_no_aki']==1
    assert result['unlabeled_patients']==1
    with Path(result['input_file']).open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert rows[0]['preoperative/age']=='5'
    assert rows[1]['aki_label']==''
    assert list(rows[0])==['patient_id','preoperative/age','preoperative/sex','aki_label','aki_stage','label_status','label_reason']
    result2=build(config,require_labels=False)
    assert result2['patients']==2
    assert 'aki_label' not in next(csv.DictReader(Path(result2['input_file']).open()))
