"""End-to-end, synthetic plotting tests; never ingest real patient records."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.preprocessing.plotting.run import run
from src.preprocessing.plotting.data import load_patient_events, load_patient_boundaries
from argparse import Namespace

PARAMETERS = '''[settings]
missing_tokens = NA
[timestamps]
fields = unixtime
[boundary.operation_start]
fields = epic.EVENTS.EVENT
phrases = start
[boundary.operation_end]
fields = epic.EVENTS.EVENT
phrases = end
[parameter.age]
phase = preoperative
source_fields = epic.DEMOGRAPHICS.age
type = numeric
unit = months
timed = false
[parameter.kreatinine]
phase = preoperative, intraoperative, postoperative
source_fields = epic.LAB.Kreatinine (umol/L)
type = timeseries
unit = umol/L
timed = true
'''
PLOTTING = '''[paths]
input_directory = results/parameters
parameter_config = config/parameters.ini
output_directory = results/plots
[general]
theme = dark
format = html
[selection]
phases = preoperative, intraoperative, postoperative
parameter_ids =
class_names =
families =
patient_ids =
mode = union
[patient_dashboards]
enabled = true
output_directory = results/plots/patients
include_plotlyjs = true
'''
HEADER = ["time_minutes","parameter_id","value","unit","type","source_field","value_origin"]

def write_csv(path:Path, rows:list[list]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file);writer.writerow(HEADER);writer.writerows(rows)

class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        config=self.root/'config';config.mkdir()
        (config/'parameters.ini').write_text(PARAMETERS,encoding='utf-8')
        (config/'plotting.ini').write_text(PLOTTING,encoding='utf-8')
        self.input=self.root/'results/parameters'
        for patient in ('ref_001','ref_002'):
            write_csv(self.input/'preoperative'/f'{patient}.csv', [
                ['', 'age','80','months','numeric','epic.DEMOGRAPHICS.age','observed'],
                [-20,'kreatinine','70','umol/L','timeseries','epic.LAB.Kreatinine (umol/L)','observed'],
            ])
            write_csv(self.input/'intraoperative'/f'{patient}.csv', [
                [0,'epic_event', 'Aankomst OK | Start anesthesie' if patient=='ref_001' else 'Start unique ref_002', '', 'str','epic.EVENTS.EVENT','observed'],
                [30,'kreatinine','75','umol/L','timeseries','epic.LAB.Kreatinine (umol/L)','observed'],
            ])
            write_csv(self.input/'postoperative'/f'{patient}.csv', [
                [180,'epic_event','Afronden zorg | Stop anesthesie' if patient=='ref_001' else 'End unique ref_002','','str','epic.EVENTS.EVENT','observed'],
                [300,'kreatinine','85','umol/L','timeseries','epic.LAB.Kreatinine (umol/L)','observed'],
            ])
        with (self.input/'summary.csv').open('w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file);writer.writerow(['patient_id','operation_start_minutes','operation_end_minutes','postoperative_end_minutes'])
            writer.writerow(['ref_001',0,180,'']);writer.writerow(['ref_002',0,'',''])

    def tearDown(self):
        self.tmp.cleanup()

    def test_operation_end_event_timeline_and_isolation(self):
        self.assertEqual(run(Namespace(config=self.root/'config/plotting.ini')),0)
        a=(self.root/'results/plots/patients/ref_001.html').read_text(encoding='utf-8').replace(r'\u00b7','·')
        b=(self.root/'results/plots/patients/ref_002.html').read_text(encoding='utf-8').replace(r'\u00b7','·')
        self.assertIn('Operation end · 180 min',a)
        self.assertIn('Operation start · 0 min',a)
        self.assertIn('Afronden zorg | Stop anesthesie',a)
        self.assertIn('Epic event timeline',a)
        self.assertIn('observed',a)
        self.assertNotIn('Start unique ref_002',a)
        self.assertNotIn('ref_002',a)
        self.assertNotIn('Operation end · 180 min',b)
        self.assertIn('Operation end is missing',b)
        self.assertNotIn('Aankomst OK | Start anesthesie',b)
        self.assertIn('plotly.js',a.lower())

    def test_metadata_and_event_loader_do_not_require_catalog(self):
        events=load_patient_events(self.input,['ref_001'])
        boundaries=load_patient_boundaries(self.input,['ref_001','ref_002'])
        self.assertEqual(len(events['ref_001']),2)
        self.assertEqual(boundaries['ref_001']['operation_end'],180)
        self.assertIsNone(boundaries['ref_002']['operation_end'])

if __name__ == '__main__':
    unittest.main()
