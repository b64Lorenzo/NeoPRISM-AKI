"""Run with: uv run pytest tests/test_multiphase.py"""
from __future__ import annotations
import csv
from pathlib import Path
from types import SimpleNamespace
from src.preprocessing.plotting.config import load_config
from src.preprocessing.plotting.selection import build_catalog, select_parameters
from src.preprocessing.plotting.data import load_observations, by_parameter
from src.preprocessing.plotting.plots import plot_distributions, plot_time_series, plot_missingness, plot_value_heatmap


def test_kreatinine_in_two_phases(tmp_path: Path):
    ini = tmp_path / 'config' / 'plotting.ini'
    ini.parent.mkdir()
    ini.write_text('''[paths]\ninput_directory = results/parameters\nparameter_config = config/parameters.ini\noutput_directory = results/plots\n[selection]\nphases = preoperative, postoperative\nparameter_ids = kreatinine\n[distributions]\nenabled = true\n[time_series]\nenabled = true\n[missingness]\nenabled = true\n[value_heatmaps]\nenabled = true\n''', encoding='utf-8')
    cfg = load_config(ini)
    param = SimpleNamespace(id='kreatinine', phases=('preoperative', 'postoperative'), source_fields=('epic.LAB.Kreatinine',), type='timeseries', unit='umol/L')
    catalog = select_parameters(build_catalog((param,), cfg), cfg)
    assert set(catalog) == {'preoperative/kreatinine', 'postoperative/kreatinine'}
    for phase, value in [('preoperative', '90'), ('postoperative', '120')]:
        path = cfg.input_dir / phase / 'ref_001.csv'
        path.parent.mkdir(parents=True)
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['time_minutes', 'parameter_id', 'value', 'unit', 'type', 'source_field'])
            writer.writeheader()
            writer.writerow(dict(time_minutes='0' if phase == 'preoperative' else '60', parameter_id='kreatinine', value=value, unit='umol/L', type='timeseries', source_field='epic.LAB.Kreatinine'))
    observations, patients = load_observations(cfg.input_dir, catalog)
    assert len(observations) == 2
    assert {k: [r.numeric_value for r in v] for k, v in by_parameter(observations).items()} == {'preoperative/kreatinine': [90.0], 'postoperative/kreatinine': [120.0]}
    assert plot_distributions(observations, catalog, cfg) == 2
    assert plot_time_series(observations, catalog, cfg) == 2
    assert plot_missingness(observations, catalog, patients, cfg) == 1
    assert plot_value_heatmap(observations, catalog, patients, cfg) == 1
    assert (cfg.output_dir / 'distributions' / 'preoperative_kreatinine.html').exists()
    assert (cfg.output_dir / 'distributions' / 'postoperative_kreatinine.html').exists()
