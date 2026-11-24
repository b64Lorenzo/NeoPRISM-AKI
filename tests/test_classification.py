import csv
import json

import pytest
from test_model_data import write_matrix


def test_shared_training_preserves_rosenblatt_artifact_and_reports(tmp_path):
    from src.models.classification import fit, predict_csv
    from src.models.rosenblatt_perceptron.execution.predict import (
        predict_csv as legacy_predict,
    )

    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix)
    output = tmp_path / "rosenblatt"
    result = fit(
        matrix,
        output,
        model_type="rosenblatt_perceptron",
        features=["preoperative/x"],
        epochs=40,
    )
    assert result["training"]["excluded_unlabeled"] == 1
    assert result["metrics"]["labels"] == [-1, 1]
    assert (output / "validation/confusion_matrix.html").is_file()
    assert (
        predict_csv(output / "model.json", matrix, tmp_path / "predictions.csv") == 25
    )
    assert legacy_predict(output / "model.json", matrix, tmp_path / "legacy.csv") == 25
    assert (tmp_path / "predictions.csv").read_bytes() == (
        tmp_path / "legacy.csv"
    ).read_bytes()
    artifact = json.loads((output / "model.json").read_text())
    assert artifact["transformer"]["features"] == ["preoperative/x"]
    assert "synthetic_" not in (output / "model.json").read_text()


def test_rosenblatt_stage_target_is_rejected_before_outputs(tmp_path):
    from src.models.classification import fit

    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix)
    with pytest.raises(ValueError, match="binary"):
        fit(
            matrix,
            tmp_path / "output",
            model_type="rosenblatt_perceptron",
            target_column="aki_stage",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("model_type", ["logistic_regression", "multilayer_perceptron"])
@pytest.mark.parametrize(
    "target_column,labels", [("aki_label", [-1, 1]), ("aki_stage", [0, 1, 2, 3])]
)
def test_probabilistic_models_train_save_and_predict_without_truth(
    tmp_path, model_type, target_column, labels
):
    from src.models.classification import fit, predict_csv

    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix)
    output = tmp_path / "trained"
    result = fit(
        matrix,
        output,
        model_type=model_type,
        target_column=target_column,
        features=["preoperative/x"],
        seed=9,
        epochs=80,
        batch_size=5,
    )
    assert result["training"]["excluded_unlabeled"] == 1
    assert result["metrics"]["labels"] == labels
    assert (
        sum(map(sum, result["metrics"]["confusion_matrix"]))
        == result["training"]["validation_patients"]
    )
    assert (output / "history.csv").is_file()
    assert (output / "training.html").is_file()
    artifact_bytes = (output / "model.json").read_bytes()
    split = json.loads((output / "split.json").read_text())
    assert not set(split["training_patient_ids"]) & set(split["validation_patient_ids"])
    novel = tmp_path / "novel.csv"
    novel.write_text("patient_id,preoperative/x\nnever_seen,2.5\n", encoding="utf8")
    destination = tmp_path / "predictions.csv"
    assert (
        predict_csv(output / "model.json", novel, destination, expected_type=model_type)
        == 1
    )
    with destination.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["patient_id"] == "never_seen"
    assert not {"aki_label", "aki_stage", "true_label"} & rows[0].keys()
    probabilities = [float(rows[0][f"probability_{label}"]) for label in labels]
    assert sum(probabilities) == pytest.approx(1)
    assert all(0 <= value <= 1 for value in probabilities)
    assert (output / "model.json").read_bytes() == artifact_bytes
    other = (
        "logistic_regression"
        if model_type == "multilayer_perceptron"
        else "multilayer_perceptron"
    )
    with pytest.raises(ValueError, match="type"):
        predict_csv(
            output / "model.json", novel, tmp_path / "wrong.csv", expected_type=other
        )
    assert not (tmp_path / "wrong.csv").exists()
