"""Unit tests: run with `python -m unittest discover -s tests` from package patch root."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from src.preprocessing.parameters.config import load_config
from src.preprocessing.parameters.derivation import derive_patient_parameters

CONFIG = '''[settings]
missing_tokens = NA
[timestamps]
fields = unixtime
[boundary.operation_start]
fields = epic.EVENTS.EVENT
phrases = start
[boundary.operation_end]
fields = epic.EVENTS.EVENT
phrases = end
[parameter.weight]
phase = preoperative
source_fields = w
type = numeric
unit = kg
timed = false
[parameter.height]
phase = preoperative
source_fields = h
type = numeric
unit = cm
timed = false
[parameter.bmi]
phase = preoperative
source_fields = bmi
type = numeric
unit = kg/m^2
timed = false
derive = bmi
weight_parameter_id = weight
height_parameter_id = height
weight_unit = kg
height_unit = cm
'''

def row(key, value, time=''):
    return {'parameter_id':key, 'value':str(value), 'time_minutes':time,
            'source_field':key, 'type':'numeric', 'unit':'', 'value_origin':'observed'}

class DerivationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'parameters.ini'
        self.path.write_text(CONFIG,encoding='utf-8')
        self.config = load_config(self.path)
    def tearDown(self): self.temp.cleanup()
    def test_derived(self):
        out = {'preoperative':[row('weight',70),row('height',175)],'intraoperative':[],'postoperative':[]}
        derive_patient_parameters(out,self.config)
        bmi = [r for r in out['preoperative'] if r['parameter_id']=='bmi']
        self.assertEqual(len(bmi),1)
        self.assertAlmostEqual(float(bmi[0]['value']),70/1.75**2)
        self.assertEqual(bmi[0]['value_origin'],'derived_bmi_from_weight_height')
    def test_preserve_recorded(self):
        out={'preoperative':[row('weight',70),row('height',175),row('bmi',21)],'intraoperative':[],'postoperative':[]}
        derive_patient_parameters(out,self.config)
        bmi=[r for r in out['preoperative'] if r['parameter_id']=='bmi']
        self.assertEqual(len(bmi),1)
        self.assertEqual(bmi[0]['value'],'21')
    def test_missing_height(self):
        out={'preoperative':[row('weight',70)],'intraoperative':[],'postoperative':[]}
        derive_patient_parameters(out,self.config)
        self.assertFalse(any(r['parameter_id']=='bmi' for r in out['preoperative']))
    def test_conflicting_static(self):
        out={'preoperative':[row('weight',70),row('weight',71),row('height',175)],'intraoperative':[],'postoperative':[]}
        derive_patient_parameters(out,self.config)
        self.assertFalse(any(r['parameter_id']=='bmi' for r in out['preoperative']))

if __name__=='__main__':unittest.main()

class FullExtractionTests(unittest.TestCase):
    def test_from_csv_and_no_postoperative_leakage(self):
        import csv
        from src.preprocessing.parameters.extractor import extract_parameters
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'parameters.ini').write_text(CONFIG, encoding='utf-8')
            data = root / 'data'
            data.mkdir()
            with (data / 'ref_001.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['unixtime','epic.EVENTS.EVENT','w','h','bmi'])
                writer.writerows([
                    [1000,'',70,175,''],
                    [1100,'start',70,175,''],
                    [1400,'end',70,175,''],
                    [1600,'',140,175,''],
                ])
            stats = extract_parameters([data],root/'out',root/'parameters.ini')
            self.assertEqual(stats['derived_readings'],1)
            with (root/'out/preoperative/ref_001.csv').open(encoding='utf-8') as f:
                rows=list(csv.DictReader(f))
            derived=[r for r in rows if r['parameter_id']=='bmi']
            self.assertEqual(len(derived),1)
            self.assertAlmostEqual(float(derived[0]['value']),70/(1.75*1.75))
            self.assertEqual(derived[0]['time_minutes'],'')
            self.assertEqual(derived[0]['value_origin'],'derived_bmi_from_weight_height')
            weights=[r for r in rows if r['parameter_id']=='weight']
            self.assertEqual({r['value'] for r in weights},{'70'})
