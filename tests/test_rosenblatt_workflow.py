"""Rosenblatt settings must reach the same course algorithm through every entry."""

import csv
import json

import pytest
from test_model_data import write_matrix

from src.cli import build_parser
from src.models.classification import fit as fit_shared
from src.models.data import Transformer, read_matrix
from src.models.rosenblatt_perceptron.core import Perceptron, train_rosenblatt
from src.models.rosenblatt_perceptron.execution.predict import load_artifact
from src.models.rosenblatt_perceptron.training import fit as fit_standalone


@pytest.mark.parametrize("fit_function", [fit_standalone, fit_shared])
@pytest.mark.parametrize("clamped", [False, True])
def test_saved_training_result_matches_reference_on_the_exact_transformed_rows(
    tmp_path, fit_function, clamped
):
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "preoperative/x", "aki_label"])
        writer.writerows([f"synthetic_{i}", i, -1 if i < 6 else 1] for i in range(12))
    output = tmp_path / "trained"
    kwargs = (
        {"model_type": "rosenblatt_perceptron"} if fit_function is fit_shared else {}
    )
    result = fit_function(matrix, output, epochs=12, clamped=clamped, seed=9, **kwargs)
    artifact = json.loads((output / "model.json").read_text())
    split = json.loads((output / "split.json").read_text())
    rows, _ = read_matrix(matrix)
    by_id = {row["patient_id"]: row for row in rows}
    training = [by_id[patient_id] for patient_id in split["training_patient_ids"]]
    transform = Transformer.from_dict(artifact["transformer"])
    examples = list(
        zip(transform.transform(training), [int(row["aki_label"]) for row in training])
    )
    weights, epochs, updates, converged, strengths = train_rosenblatt(
        examples, 12, 0.0, clamped
    )
    assert artifact["weights"] + ([artifact["bias"]] if clamped else []) == weights
    assert clamped or artifact["bias"] == 0.0
    assert result["training"]["clamped"] is clamped
    assert result["training"]["learning_rate"] == 1.0 / len(weights)
    assert result["training"]["strengths"] == strengths
    assert result["training"]["epochs_run"] == epochs
    assert result["training"]["updates"] == updates
    assert result["training"]["converged"] == converged
    loaded, _ = load_artifact(output / "model.json")
    assert loaded == Perceptron(
        artifact["weights"], artifact["bias"], artifact["threshold"]
    )
    with (output / "plots/convergence/epoch_history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == epochs
    nonpositive = sum(label * loaded.score(inputs) <= 0.0 for inputs, label in examples)
    assert int(history[-1]["nonpositive_margins"]) == nonpositive
    assert (nonpositive == 0) is converged


def test_cli_exposes_clamped_but_no_independent_learning_rate(capsys):
    args = build_parser().parse_args(
        ["rosenblatt-train", "--clamped", "--threshold", "-1"]
    )
    assert args.clamped is True
    assert args.threshold == -1.0
    assert build_parser().parse_args(["rosenblatt-train"]).clamped is False
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["rosenblatt-train", "--learning-rate", "0.1"])
    assert error.value.code == 2
    assert "--learning-rate" in capsys.readouterr().err


def test_shared_rosenblatt_rejects_a_custom_rate_without_creating_outputs(tmp_path):
    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix)
    with pytest.raises(ValueError, match="learning.rate|fixed"):
        fit_shared(
            matrix,
            tmp_path / "trained",
            model_type="rosenblatt_perceptron",
            learning_rate=0.1,
        )
    assert not (tmp_path / "trained").exists()


@pytest.mark.parametrize("clamped", [False, True])
def test_real_cli_and_xml_train_the_same_course_model(tmp_path, clamped):
    from test_model_cli_smoke import _run

    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "preoperative/x", "aki_label"])
        writer.writerows([f"synthetic_{i}", i, -1 if i < 6 else 1] for i in range(12))
    cli_output = tmp_path / "cli"
    args = ["--clamped"] if clamped else []
    _run(
        "rosenblatt-train",
        "--input",
        matrix,
        "--output",
        cli_output,
        "--max-epochs",
        "12",
        "--seed",
        "9",
        *args,
    )
    xml = tmp_path / "pipeline.xml"
    xml.write_text(
        f'''<pipelineConfiguration><models>
<model id="reference" type="rosenblatt_perceptron" run="true">
<inputMatrix>matrix.csv</inputMatrix><outputDir>xml</outputDir>
<targetColumn>aki_label</targetColumn><features><column>preoperative/x</column></features>
<split trainFraction="0.8" seed="9"/><options>
<option name="epochs" value="12"/><option name="clamped" value="{str(clamped).lower()}"/>
</options></model></models></pipelineConfiguration>''',
        encoding="utf8",
    )
    _run("pipeline", "--config", xml)
    cli_artifact = json.loads((cli_output / "model.json").read_text())
    xml_artifact = json.loads((tmp_path / "xml/model.json").read_text())
    assert cli_artifact == xml_artifact
    assert cli_artifact["training"]["clamped"] is clamped
    assert cli_artifact["training"]["training_order"] == "fixed"
    assert cli_artifact["training"]["updates"] == sum(
        cli_artifact["training"]["strengths"]
    )
    novel = tmp_path / "novel.csv"
    novel.write_text("patient_id,preoperative/x\nnew_synthetic,3\n", encoding="utf8")
    prediction = tmp_path / "prediction.csv"
    _run(
        "rosenblatt-predict",
        "--model",
        cli_output / "model.json",
        "--input",
        novel,
        "--output",
        prediction,
    )
    with prediction.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["patient_id"] == "new_synthetic"
    assert "true_label" not in rows[0]


def test_legacy_version_one_artifact_keeps_its_stored_bias(tmp_path):
    transform = Transformer.fit([{"x": "-1"}, {"x": "1"}], ["x"])
    artifact = {
        "schema_version": 1,
        "model_type": "binary_rosenblatt_perceptron",
        "weights": [0.25, 0.0],
        "bias": 2.0,
        "threshold": 3.0,
        "transformer": transform.to_dict(),
        "training": {"seed": 2026},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(artifact), encoding="utf8")
    model, loaded_transform = load_artifact(path)
    assert model == Perceptron([0.25, 0.0], 2.0, 3.0)
    assert model.score(loaded_transform.transform_one({"x": "0"})) == 2.0
