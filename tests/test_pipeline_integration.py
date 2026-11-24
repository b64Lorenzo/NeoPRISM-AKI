"""Exercise XML -> extraction -> matrix -> all model families on synthetic CSVs."""

import csv
import json

from test_model_cli_smoke import _run


def test_xml_cli_extracts_builds_and_trains_all_model_families(tmp_path):
    config = tmp_path / "config"
    raw = tmp_path / "raw"
    config.mkdir()
    raw.mkdir()
    parameters = config / "parameters.ini"
    parameters.write_text(
        """[settings]
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
source_fields = age
unit = months
type = numeric
timed = false
""",
        encoding="utf8",
    )
    inputs = config / "inputs.ini"
    inputs.write_text(
        """[paths]
input_directory = extracted
parameter_config = config/parameters.ini
labels_csv = labels.csv
output_directory = matrices
[features]
parameter_ids = preoperative/age
aggregation = patient_mean
[labels]
source = external
kind = kdigo_stage
column = stage
""",
        encoding="utf8",
    )
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "stage"])
        for index in range(16):
            patient_id = f"syntheticA{index:02}"
            writer.writerow([patient_id, index // 4])
            (raw / f"{patient_id}.csv").write_text(
                f"timestamp,event,age\n2026-01-01T09:00:00,,{index + 1}\n"
                f"2026-01-01T10:00:00,start,{index + 1}\n"
                f"2026-01-01T11:00:00,end,{index + 1}\n",
                encoding="utf8",
            )
    entries = []
    for kind, target in [
        ("rosenblatt_perceptron", "aki_label"),
        ("logistic_regression", "aki_stage"),
        ("multilayer_perceptron", "aki_label"),
    ]:
        entries.append(f'''<model id="{kind}" type="{kind}" run="true">
<inputMatrix>../matrices/patients_labeled.csv</inputMatrix>
<outputDir>../models/{kind}</outputDir>
<targetColumn>{target}</targetColumn>
<features><column>preoperative/age</column></features>
<split trainFraction="0.75" seed="2026"/>
<options><option name="epochs" value="12"/></options>
</model>''')
    pipeline = config / "pipeline.xml"
    pipeline.write_text(
        """<pipelineConfiguration>
<preprocessing>
<extraction run="true" config="parameters.ini" outputDir="../extracted"><input>../raw</input></extraction>
<inputsModel run="true" config="inputs.ini" unlabeled="false"/>
</preprocessing>
<models>"""
        + "".join(entries)
        + "</models></pipelineConfiguration>",
        encoding="utf8",
    )
    _run("pipeline", "--config", pipeline)
    with (tmp_path / "matrices/patients_labeled.csv").open(newline="") as stream:
        matrix_rows = list(csv.DictReader(stream))
    assert len(matrix_rows) == 16
    assert {row["aki_stage"] for row in matrix_rows} == {"0", "1", "2", "3"}
    for kind, target in [
        ("rosenblatt_perceptron", "aki_label"),
        ("logistic_regression", "aki_stage"),
        ("multilayer_perceptron", "aki_label"),
    ]:
        directory = tmp_path / "models" / kind
        artifact = json.loads((directory / "model.json").read_text())
        assert (
            artifact["training"]["training_patients"]
            + artifact["training"]["validation_patients"]
            == 16
        )
        assert artifact["transformer"]["features"] == ["preoperative/age"]
        metrics = json.loads((directory / "validation/metrics.json").read_text())
        assert metrics["labels"] == ([0, 1, 2, 3] if target == "aki_stage" else [-1, 1])
        assert (
            sum(map(sum, metrics["confusion_matrix"]))
            == artifact["training"]["validation_patients"]
        )
        assert (directory / "validation/confusion_matrix.html").is_file()
