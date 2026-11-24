import csv
import json

import numpy as np
import pytest


def test_confusion_matrix_has_explicit_order_and_absent_classes():
    from src.models.evaluation.metrics import confusion_matrix

    matrix = confusion_matrix([2, 0, 2, 1], [1, 0, 2, 0], labels=[2, 0, 1, 3])
    assert matrix == [[1, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    assert confusion_matrix(np.array([-1, 1]), np.array([1, 1]), labels=[-1, 1]) == [
        [0, 1],
        [0, 1],
    ]


@pytest.mark.parametrize(
    "truth,predicted,labels",
    [
        ([], [], [0, 1]),
        ([0], [], [0, 1]),
        ([0], [2], [0, 1]),
        ([0], [0], [0, 0]),
        ([None], [0], [0, 1]),
    ],
)
def test_confusion_matrix_rejects_ambiguous_or_invalid_inputs(truth, predicted, labels):
    from src.models.evaluation.metrics import confusion_matrix

    with pytest.raises(ValueError):
        confusion_matrix(truth, predicted, labels=labels)


def test_multiclass_report_preserves_undefined_metrics(tmp_path):
    from src.models.evaluation.report import save_classification_report

    result = save_classification_report(
        tmp_path,
        [0, 0, 1, 2],
        [0, 1, 1, 1],
        labels=[0, 1, 2, 3],
    )
    assert result["accuracy"] == 0.5
    assert result["per_class"][3]["recall"] is None
    assert result["per_class"][2]["precision"] is None
    assert json.loads((tmp_path / "metrics.json").read_text()) == result
    with (tmp_path / "confusion_matrix.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        ["true/predicted", "0", "1", "2", "3"],
        ["0", "1", "1", "0", "0"],
        ["1", "0", "1", "0", "0"],
        ["2", "0", "1", "0", "0"],
        ["3", "0", "0", "0", "0"],
    ]
    assert (tmp_path / "confusion_matrix.html").is_file()


def test_existing_binary_metrics_keep_their_contract():
    from src.models.rosenblatt_perceptron.validation.metrics import evaluate

    metrics = evaluate([-1, -1, 1, 1], [-1, 1, -1, 1], [-2, 1, -1, 2])
    assert metrics["labels"] == [-1, 1]
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["tn"] == metrics["fp"] == metrics["fn"] == metrics["tp"] == 1
    assert metrics["roc_auc"] == 0.75
