"""Real subprocess CLI/XML checks with synthetic matrices and scratch-only outputs."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KINDS = [
    ("mlp", "multilayer_perceptron"),
    ("multioutput", "multi_output_perceptron"),
    ("logistic", "logistic_regression"),
]


def _inputs(root):
    matrix = root / "synthetic matrix.csv"
    with matrix.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "preoperative/x", "aki_stage"])
        writer.writerows([f"synthetic_{i}", i, i // 4] for i in range(16))
    settings = root / "settings with spaces"
    settings.mkdir()
    ini = settings / "models.ini"
    ini.write_text(
        """[ray]
num_cpus = 1
cpus_per_task = 1
timeout_seconds = 60
[multilayer_perceptron]
model_type = multilayer_perceptron
hidden_layers = 5,3
hidden_activations = relu,tanh
init_methods = he,xavier,xavier
epochs = 3
[multi_output_perceptron]
model_type = multi_output_perceptron
epochs = 3
[logistic_regression]
model_type = logistic_regression
epochs = 3
""",
        encoding="utf8",
    )
    return matrix, ini


def _command(root, *arguments):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", *map(str, arguments)],
        cwd=root,
        check=False,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _verify(output, kind):
    artifact = json.loads((output / "model.json").read_text())
    assert artifact["model_type"] == kind
    execution = artifact["training"]["execution"]
    assert execution["backend"] == "ray"
    assert execution["worker_pid"] != execution["driver_pid"]
    assert execution["worker_id"]
    assert execution["num_cpus"] == execution["cpus_per_task"] == 1
    if kind == "multilayer_perceptron":
        assert artifact["model"]["shape"] == [2, 5, 3, 4]
        assert artifact["model"]["activation_functions"] == ["relu", "tanh", "softmax"]
    with (output / "validation/confusion_matrix.csv").open(newline="") as stream:
        table = list(csv.reader(stream))
    split = json.loads((output / "split.json").read_text())
    assert len(table) == 5
    assert sum(int(value) for row in table[1:] for value in row[1:]) == len(
        split["validation_patient_ids"]
    )
    return split


@pytest.mark.parametrize("prefix,kind", KINDS)
def test_actual_configured_training_and_prediction_cli(tmp_path, prefix, kind):
    matrix, ini = _inputs(tmp_path)
    output = tmp_path / "trained model"
    _command(
        tmp_path,
        prefix + "-train",
        "--input",
        matrix,
        "--output",
        output,
        "--config",
        ini,
        "--target-column",
        "aki_stage",
    )
    _verify(output, kind)
    novel = tmp_path / "novel.csv"
    novel.write_text("patient_id,preoperative/x\nnew,4\n", encoding="utf8")
    prediction = tmp_path / "predicted.csv"
    _command(
        tmp_path,
        prefix + "-predict",
        "--model",
        output / "model.json",
        "--input",
        novel,
        "--output",
        prediction,
    )
    with prediction.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 1
    assert "true_label" not in records[0]
    key = "score_" if kind == "multi_output_perceptron" else "probability_"
    assert len([field for field in records[0] if field.startswith(key)]) == 4


def test_actual_xml_runs_all_three_ini_profiles_with_identical_splits(tmp_path):
    matrix, ini = _inputs(tmp_path)
    models = "".join(
        f'''<model id="{prefix}" type="{kind}" run="true">
      <inputMatrix>{matrix.name}</inputMatrix><outputDir>{prefix}</outputDir>
      <targetColumn>aki_stage</targetColumn><features><column>preoperative/x</column></features>
      <split trainFraction="0.8" seed="2026"/>
      <options><option name="config" value="{ini.relative_to(tmp_path).as_posix()}"/></options>
    </model>'''
        for prefix, kind in KINDS
    )
    xml = tmp_path / "synthetic.xml"
    xml.write_text(
        "<pipelineConfiguration><models>"
        + models
        + "</models></pipelineConfiguration>",
        encoding="utf8",
    )
    _command(tmp_path, "pipeline", "--config", xml)
    splits = [_verify(tmp_path / prefix, kind) for prefix, kind in KINDS]
    assert all(split == splits[0] for split in splits)
