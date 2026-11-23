from pathlib import Path
import pytest
from src.preprocessing.plotting.config import load_config
from src.preprocessing.plotting.selection import ParameterInfo, select_heatmap_parameters


def catalog():
    return {f'preoperative/{name}': ParameterInfo(name, 'preoperative', kind, '', (), (), ())
            for name, kind in [('age', 'numeric'), ('sex', 'str'), ('bmi', 'numeric')]}


def test_independent_config(tmp_path: Path):
    path = tmp_path / 'config' / 'plotting.ini'
    path.parent.mkdir()
    path.write_text('''[paths]
input_directory = results/parameters
parameter_config = config/parameters.ini
output_directory = results/plots
[selection]
parameter_ids = age
[missingness]
enabled = true
parameter_ids = sex, bmi
[value_heatmaps]
enabled = true
parameter_ids = bmi
''')
    cfg = load_config(path)
    assert cfg.selection.parameter_ids == ('age',)
    assert cfg.missingness_parameter_ids == ('sex', 'bmi')
    assert cfg.heatmap_parameter_ids == ('bmi',)


def test_missingness_includes_categorical():
    selected = select_heatmap_parameters(catalog(), ('preoperative',), ('sex', 'bmi'))
    assert list(selected) == ['preoperative/sex', 'preoperative/bmi']


def test_numeric_heatmap_excludes_categorical():
    selected = select_heatmap_parameters(catalog(), ('preoperative',), ('all',), numeric_only=True)
    assert 'preoperative/sex' not in selected


@pytest.mark.parametrize('ids', [(), ('all', 'bmi'), ('unknown',)])
def test_invalid_selection(ids):
    with pytest.raises(ValueError):
        select_heatmap_parameters(catalog(), ('preoperative',), ids)
