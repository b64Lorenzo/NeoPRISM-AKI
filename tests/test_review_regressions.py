import json

import pytest

from src.models.evaluation.metrics import evaluate_binary
from src.models.logistic_regression import LogisticRegression
from src.models.neural_network import MultiLayerPerceptron


def test_binary_metrics_accept_single_pass_iterators():
    result = evaluate_binary(iter([-1, 1]), iter([-1, 1]), scores=iter([0.0, 1.0]))
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["roc_auc"] == 1.0


@pytest.mark.parametrize("model_class", [LogisticRegression, MultiLayerPerceptron])
def test_string_class_identity_survives_numpy_and_json_round_trips(model_class):
    labels = ["a", "a\0", "b"]
    model = (
        model_class([1, 3]) if model_class is MultiLayerPerceptron else model_class()
    )
    model.fit([[0.0], [1.0], [2.0]], labels, epochs=1)
    assert model.classes_.tolist() == labels
    payload = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = model_class.from_dict(payload)
    assert restored.classes_.tolist() == labels
    assert restored.predict_proba([[1.0]]).shape == (1, 3)
