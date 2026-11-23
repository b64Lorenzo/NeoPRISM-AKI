import pytest
from src.preprocessing.plotting.config import load_config
from src.preprocessing.plotting.selection import ParameterInfo, select_heatmap_parameters


def config_file(tmp_path, phases=''):
    f=tmp_path/'config'/'plotting.ini';f.parent.mkdir()
    f.write_text('''[paths]
input_directory = results/parameters
parameter_config = config/parameters.ini
output_directory = results/plots
[selection]
phases = preoperative
parameter_ids = bmi
[missingness]
enabled = true
parameter_ids = sex
'''+phases+'''[parameter_availability]
enabled = true
parameter_ids = bmi
phases = postoperative
[value_heatmaps]
enabled = true
parameter_ids = bmi
phases = all
''')
    return f


def test_independent_phases(tmp_path):
    c=load_config(config_file(tmp_path,'phases = intraoperative, postoperative\n'))
    assert c.selection.phases == ('preoperative',)
    assert c.missingness_phases == ('intraoperative','postoperative')
    assert c.availability_phases == ('postoperative',)
    assert set(c.heatmap_phases)=={'preoperative','intraoperative','postoperative'}


def test_default_all_phases(tmp_path):
    c=load_config(config_file(tmp_path))
    assert len(c.missingness_phases)==3


@pytest.mark.parametrize('phases',['all, preoperative','unknown','preoperative, preoperative'])
def test_invalid_phases(tmp_path, phases):
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path,f'phases = {phases}\n'))


def test_selected_parameter_phase_pairs():
    catalog={f'{phase}/bmi':ParameterInfo('bmi',phase,'numeric','',(),(),()) for phase in ('preoperative','postoperative')}
    selected=select_heatmap_parameters(catalog,('postoperative',),('bmi',))
    assert list(selected)==['postoperative/bmi']
