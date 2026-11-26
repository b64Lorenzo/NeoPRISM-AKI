import csv
import json

import pytest

from src.models.classification import fit


def _matrix(path):
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "preoperative/x", "aki_stage"])
        writer.writerows([f"synthetic_{i}", i, i // 4] for i in range(16))
    return path


def test_patient_workflow_preserves_a_different_activation_for_each_hidden_layer(
    tmp_path,
):
    output = tmp_path / "trained"
    fit(
        _matrix(tmp_path / "matrix.csv"),
        output,
        model_type="multilayer_perceptron",
        target_column="aki_stage",
        hidden_layers=[8, 4],
        hidden_activations=["relu", "tanh"],
        init_methods=["he", "xavier", "xavier"],
        epochs=3,
    )
    payload = json.loads((output / "model.json").read_text())["model"]
    assert payload["shape"] == [2, 8, 4, 4]
    assert payload["activation_functions"] == ["relu", "tanh", "softmax"]
    assert payload["init_methods"] == ["he", "xavier", "xavier"]


@pytest.mark.parametrize(
    "options",
    [
        {"hidden_layers": [8, 4], "hidden_activations": ["relu"]},
        {"hidden_layers": [8], "hidden_activations": ["relu"], "activation": "relu"},
    ],
)
def test_ambiguous_or_mismatched_activations_fail_before_outputs(tmp_path, options):
    output = tmp_path / "trained"
    with pytest.raises(ValueError, match="activation"):
        fit(
            _matrix(tmp_path / "matrix.csv"),
            output,
            model_type="multilayer_perceptron",
            target_column="aki_stage",
            epochs=1,
            **options,
        )
    assert not output.exists()
