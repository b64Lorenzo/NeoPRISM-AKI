from src.preprocessing.parameters.config import Parameter, ExtractionConfig
from src.preprocessing.parameters.derivation import derive_patient_parameters, get_derivation


def param(id, phase, derive, **kwargs):
    return Parameter(id=id, phases=(phase,), source_fields=(), type="numeric", unit=kwargs.pop("unit", ""),
                     timed=False, derive=derive, **kwargs)


def config(*parameters):
    return ExtractionConfig(tuple(parameters), frozenset({""}), False, ("unixtime",), {})


def test_bmi_and_observed_precedence():
    rows = {"preoperative": [
        {"parameter_id": "weight", "value": "70", "time_minutes": ""},
        {"parameter_id": "height", "value": "175", "time_minutes": ""},
    ], "intraoperative": [], "postoperative": []}
    bmi = param("bmi", "preoperative", "bmi", unit="kg/m^2")
    derive_patient_parameters(rows, config(bmi))
    result = next(r for r in rows["preoperative"] if r["parameter_id"] == "bmi")
    assert abs(float(result["value"]) - 22.8571428571) < 1e-7
    assert result["value_origin"] == "derived_bmi_from_weight_height"
    derive_patient_parameters(rows, config(bmi))
    assert sum(r["parameter_id"] == "bmi" for r in rows["preoperative"]) == 1


def test_operation_duration_and_missing_boundaries():
    duration = param("operation_duration", "intraoperative", "operation_duration", unit="minutes")
    rows = {"preoperative": [], "intraoperative": [], "postoperative": []}
    derive_patient_parameters(rows, config(duration), {"operation_start": 1000, "operation_end": 11800})
    assert float(rows["intraoperative"][0]["value"]) == 180
    assert rows["intraoperative"][0]["value_origin"] == "derived_operation_duration"
    empty = {"preoperative": [], "intraoperative": [], "postoperative": []}
    derive_patient_parameters(empty, config(duration), {"operation_start": 1000, "operation_end": None})
    assert empty["intraoperative"] == []


def test_unknown_derivation():
    import pytest
    with pytest.raises(ValueError, match="Unknown derivation"):
        get_derivation("unregistered")


def test_duration_config_accepts_empty_source_fields(tmp_path):
    from src.preprocessing.parameters.config import load_config
    ini = tmp_path / "parameters.ini"
    ini.write_text('''[settings]
missing_tokens = NA
[timestamps]
fields = unixtime
[boundary.operation_start]
fields = epic.EVENTS.EVENT
phrases = start
[boundary.operation_end]
fields = epic.EVENTS.EVENT
phrases = end
[parameter.operation_duration]
phase = intraoperative
source_fields =
type = numeric
unit = minutes
timed = false
derive = operation_duration
''')
    loaded = load_config(ini)
    assert loaded.parameters[0].derive == "operation_duration"
