from pathlib import Path
from src.preprocessing.plotting.config import load_config


def test_heatmap_specific_ids(tmp_path: Path):
    cfg = tmp_path / 'config' / 'plotting.ini'
    cfg.parent.mkdir()
    cfg.write_text('''[paths]
input_directory = results/parameters
parameter_config = config/parameters.ini
output_directory = results/plots
[selection]
parameter_ids = sex
[value_heatmaps]
enabled = true
parameter_ids = bmi, kreatinine
''')
    parsed = load_config(cfg)
    assert parsed.selection.parameter_ids == ('sex',)
    assert parsed.heatmap_parameter_ids == ('bmi', 'kreatinine')
