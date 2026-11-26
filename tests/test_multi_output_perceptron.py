"""Synthetic tests for the sequential, score-only multiclass perceptron."""

import json

import numpy as np
import pytest

from src.models.multi_output_perceptron import MultiOutputPerceptron


def test_one_epoch_matches_seeded_sequential_mistake_updates():
    X = np.array([[1.0, 0.5], [-0.5, 1.0], [0.2, -0.3], [1.5, -0.7]])
    y = np.array([10, 30, -7, 10])
    classes, encoded = np.unique(y, return_inverse=True)
    weights, bias = np.zeros((3, 2)), np.zeros(3)
    updates = 0
    for row in np.random.default_rng(7).permutation(len(X)):
        predicted = np.argmax(weights @ X[row] + bias)
        actual = encoded[row]
        if predicted != actual:
            weights[actual] += 0.2 * X[row]
            weights[predicted] -= 0.2 * X[row]
            bias[actual] += 0.2
            bias[predicted] -= 0.2
            updates += 1
    scores = X @ weights.T + bias
    predictions = classes[np.argmax(scores, axis=1)]
    accuracy = np.mean(predictions == y)

    model = MultiOutputPerceptron(seed=7)
    assert model.fit(X, y, epochs=1, learning_rate=0.2) is model
    np.testing.assert_allclose(model.weights_, weights, rtol=0, atol=0)
    np.testing.assert_allclose(model.bias_, bias, rtol=0, atol=0)
    np.testing.assert_array_equal(model.classes_, classes)
    np.testing.assert_allclose(model.decision_function(X), scores)
    np.testing.assert_array_equal(model.predict(X), predictions)
    np.testing.assert_allclose(model.weights_.sum(axis=0), 0, atol=1e-15)
    assert model.bias_.sum() == pytest.approx(0, abs=1e-15)
    assert model.n_features_in_ == 2
    assert model.updates_ == updates
    assert model.converged_ is False
    assert model.history_ == [
        {"epoch": 1, "loss": 1.0 - accuracy, "accuracy": accuracy}
    ]
    assert not hasattr(model, "predict_proba")


def test_tied_scores_choose_first_sorted_class_and_correct_ties_do_not_update():
    # Seed 1 visits labels in this order. The first correct tie does not update;
    # the second sample updates only the true and predicted bias rows.
    model = MultiOutputPerceptron(seed=1).fit(
        [[0.0], [0.0], [0.0]], [10, 20, 30], epochs=1, learning_rate=0.2
    )
    assert model.updates_ == 2
    np.testing.assert_array_equal(model.weights_, np.zeros((3, 1)))
    np.testing.assert_allclose(model.bias_, [-0.2, 0.0, 0.2])
    model.bias_[:] = 0
    np.testing.assert_array_equal(model.predict([[0.0]]), [10])


@pytest.mark.parametrize(
    "X,y",
    [
        ([], []),
        ([[1.0]], [0]),
        ([[1.0], [2.0]], [0]),
        ([[1.0], [2.0]], [[0], [1]]),
        ([[1.0], [2.0]], [0.5, 1.5]),
        ([[1.0], [2.0]], [False, True]),
        ([[1.0], [2.0]], [1, "one"]),
        ([[1.0], [2.0]], [None, 1]),
        ([[1.0], [2.0]], [0, 2**70]),
        ([[1.0], [float("nan")]], [0, 1]),
        ([[1.0], [float("inf")]], [0, 1]),
        ([[1.0 + 1j], [2.0]], [0, 1]),
        ([["one"], ["two"]], [0, 1]),
        ([[1.0], [2.0, 3.0]], [0, 1]),
        (np.empty((2, 0)), [0, 1]),
    ],
)
def test_invalid_training_data_is_rejected(X, y):
    with pytest.raises(ValueError):
        MultiOutputPerceptron().fit(X, y, epochs=1)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "7"])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError):
        MultiOutputPerceptron(seed=seed)


@pytest.mark.parametrize(
    "arguments",
    [
        {"epochs": 0},
        {"epochs": True},
        {"epochs": 1.5},
        {"learning_rate": 0},
        {"learning_rate": -1},
        {"learning_rate": True},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
    ],
)
def test_invalid_fit_hyperparameters_are_rejected(arguments):
    with pytest.raises(ValueError):
        MultiOutputPerceptron().fit([[0.0], [1.0]], [-1, 1], **arguments)


def test_zero_postepoch_error_waits_for_zero_online_mistakes():
    X, y = [[-1.0], [1.0]], [-1, 1]
    first = MultiOutputPerceptron().fit(X, y, epochs=1)
    assert first.history_ == [{"epoch": 1, "loss": 0.0, "accuracy": 1.0}]
    assert first.updates_ == 1
    assert first.converged_ is False
    complete = MultiOutputPerceptron().fit(X, y, epochs=20)
    assert complete.converged_ is True
    assert complete.updates_ == 1
    assert len(complete.history_) == 2
    assert complete.history_[-1] == {"epoch": 2, "loss": 0.0, "accuracy": 1.0}
    np.testing.assert_array_equal(complete.weights_, first.weights_)
    np.testing.assert_array_equal(complete.bias_, first.bias_)


def test_separable_multiclass_data_converges():
    X = [[2.0, 0.0], [0.0, 2.0], [-2.0, -2.0]]
    y = [10, 30, -7]
    model = MultiOutputPerceptron(seed=4).fit(X, y, epochs=50)
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.converged_ is True
    assert 1 < len(model.history_) < 50
    assert model.history_[-1]["loss"] == 0.0


def test_nonseparable_data_uses_new_seeded_permutation_each_epoch():
    X = np.zeros((6, 1))
    y = np.array([0, 1, 2, 0, 1, 2])
    weights, bias = np.zeros((3, 1)), np.zeros(3)
    history, updates = [], 0
    rng = np.random.default_rng(9)
    for epoch in range(1, 6):
        for row in rng.permutation(len(X)):
            predicted = np.argmax(bias)
            if predicted != y[row]:
                bias[y[row]] += 0.3
                bias[predicted] -= 0.3
                updates += 1
        accuracy = float(np.mean(np.argmax(bias) == y))
        history.append({"epoch": epoch, "loss": 1.0 - accuracy, "accuracy": accuracy})
    model = MultiOutputPerceptron(seed=9).fit(X, y, epochs=5, learning_rate=0.3)
    np.testing.assert_array_equal(model.weights_, weights)
    np.testing.assert_array_equal(model.bias_, bias)
    assert model.history_ == history
    assert model.updates_ == updates
    assert model.converged_ is False
    assert len(model.history_) == 5


@pytest.mark.parametrize("y", [[10, -7, 30], ["a", "a\0", "b"]])
def test_json_roundtrip_preserves_scores_labels_history_and_statistics(y):
    X = [[-1.0], [0.0], [1.0]]
    model = MultiOutputPerceptron(seed=7).fit(X, y, epochs=5)
    payload = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    assert set(payload) == {
        "version",
        "model_type",
        "seed",
        "n_features_in",
        "classes",
        "weights",
        "bias",
        "history",
        "updates",
        "converged",
    }
    assert payload["version"] == 1
    assert payload["model_type"] == "multi_output_perceptron"
    assert payload["updates"] == model.updates_
    assert payload["converged"] is model.converged_
    restored = MultiOutputPerceptron.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.classes_.tolist() == sorted(y)
    if isinstance(y[0], str):
        assert restored.classes_.dtype == object
    np.testing.assert_allclose(
        restored.decision_function(X), model.decision_function(X)
    )
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.history_ == model.history_
    assert restored.updates_ == model.updates_
    assert restored.converged_ is model.converged_


def test_json_roundtrip_preserves_converged_state():
    model = MultiOutputPerceptron().fit([[-1.0], [1.0]], [-1, 1])
    restored = MultiOutputPerceptron.from_dict(model.to_dict())
    assert restored.converged_ is True
    assert restored.updates_ == 1
    assert len(restored.history_) == 2


def test_serialization_detaches_every_mutable_container():
    model = MultiOutputPerceptron().fit([[-1.0], [1.0]], [-1, 1], epochs=1)
    payload = model.to_dict()
    snapshot = json.loads(json.dumps(payload))
    restored = MultiOutputPerceptron.from_dict(payload)
    payload["weights"][0][0] += 1
    payload["bias"][0] += 1
    payload["classes"][0] = -9
    payload["history"][0]["accuracy"] = 0.5
    assert model.to_dict() == snapshot
    assert restored.to_dict() == snapshot
    restored.weights_[0, 0] += 1
    restored.bias_[0] += 1
    restored.classes_[0] = -9
    restored.history_[0]["accuracy"] = 0.5
    assert model.to_dict() == snapshot
    assert payload["weights"][0][0] == snapshot["weights"][0][0] + 1


def test_fit_is_fresh_deterministic_readonly_and_has_no_implicit_io(monkeypatch):
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array(["a", "a\0", "b", "a"], dtype=object)
    original_X, original_y = X.copy(), y.copy()
    X.setflags(write=False)
    y.setflags(write=False)
    state = np.random.get_state()
    model = MultiOutputPerceptron(seed=17)

    def forbid_io(*args, **kwargs):
        raise AssertionError("Numerical cores must not open files")

    with monkeypatch.context() as context:
        context.setattr("builtins.open", forbid_io)
        model.fit(X, y, epochs=5)
        first = model.to_dict()
        model.fit(X, y, epochs=5)
        assert model.to_dict() == first
        restored = MultiOutputPerceptron.from_dict(first)
        restored.predict(X)
        restored.decision_function(X)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


def test_prediction_requires_fitted_model_and_valid_feature_width():
    model = MultiOutputPerceptron()
    with pytest.raises(ValueError, match="fitted"):
        model.decision_function([[1.0]])
    with pytest.raises(ValueError, match="fitted"):
        model.predict([[1.0]])
    with pytest.raises(ValueError, match="fitted"):
        model.to_dict()
    model.fit([[-1.0], [1.0]], [-1, 1], epochs=1)
    for X in ([1.0], [[1.0, 2.0]], [[float("nan")]], [[float("inf")]]):
        with pytest.raises(ValueError):
            model.decision_function(X)
        with pytest.raises(ValueError):
            model.predict(X)
    assert model.decision_function(np.empty((0, 1))).shape == (0, 2)
    assert model.predict(np.empty((0, 1))).shape == (0,)


@pytest.fixture
def perceptron_payload():
    return MultiOutputPerceptron().fit([[0.0], [0.0]], [-1, 1], epochs=3).to_dict()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", True),
        ("model_type", "unknown"),
        ("seed", -1),
        ("seed", True),
        ("n_features_in", 2),
        ("n_features_in", True),
        ("classes", [-1, -1]),
        ("classes", [1, -1]),
        ("classes", [-1]),
        ("classes", [-1.0, 1.0]),
        ("classes", [False, True]),
        ("classes", [1, "positive"]),
        ("classes", [-1, 2**70]),
        ("weights", []),
        ("weights", [[True], [0.0]]),
        ("weights", [[float("nan")], [0.0]]),
        ("weights", [[0.0], [float("inf")]]),
        ("bias", [[0.0, 0.0]]),
        ("bias", [True, 0.0]),
        ("history", []),
        ("history", {}),
        ("history", [{"epoch": 0, "loss": 0.5, "accuracy": 0.5}]),
        ("history", [{"epoch": True, "loss": 0.5, "accuracy": 0.5}]),
        ("history", [{"epoch": 1, "loss": -0.1, "accuracy": 0.5}]),
        ("history", [{"epoch": 1, "loss": 1.1, "accuracy": 0.5}]),
        ("history", [{"epoch": 1, "loss": 0.2, "accuracy": 0.5}]),
        ("history", [{"epoch": 1, "loss": float("nan"), "accuracy": 0.5}]),
        ("history", [{"epoch": 1, "loss": 0.5, "accuracy": 1.1}]),
        ("history", [{"epoch": 1, "loss": 0.5, "accuracy": None}]),
        ("history", [{"epoch": 1, "loss": 0.5}]),
        ("history", [{"epoch": 1, "loss": 0.5, "accuracy": 0.5, "extra": 1}]),
        ("updates", -1),
        ("updates", True),
        ("updates", 0.5),
        ("updates", 0),
        ("updates", 2),
        ("converged", 1),
        ("converged", "false"),
        ("converged", None),
        ("converged", True),
    ],
)
def test_malformed_saved_metadata_is_rejected(perceptron_payload, field, value):
    perceptron_payload[field] = value
    with pytest.raises(ValueError):
        MultiOutputPerceptron.from_dict(perceptron_payload)


@pytest.mark.parametrize(
    "field",
    [
        "version",
        "model_type",
        "seed",
        "n_features_in",
        "classes",
        "weights",
        "bias",
        "history",
        "updates",
        "converged",
    ],
)
def test_missing_payload_fields_are_rejected(perceptron_payload, field):
    del perceptron_payload[field]
    with pytest.raises(ValueError):
        MultiOutputPerceptron.from_dict(perceptron_payload)


@pytest.mark.parametrize("payload", [[], None, {}, "model"])
def test_non_model_payload_is_rejected(payload):
    with pytest.raises(ValueError):
        MultiOutputPerceptron.from_dict(payload)


def test_unknown_fields_and_non_json_values_are_rejected(perceptron_payload):
    perceptron_payload["unexpected"] = 1
    with pytest.raises(ValueError):
        MultiOutputPerceptron.from_dict(perceptron_payload)
    del perceptron_payload["unexpected"]
    perceptron_payload["history"][0]["loss"] = object()
    with pytest.raises(ValueError):
        MultiOutputPerceptron.from_dict(perceptron_payload)


def test_first_epoch_cannot_claim_online_convergence():
    payload = MultiOutputPerceptron().fit([[-1.0], [1.0]], [-1, 1], epochs=1).to_dict()
    payload["converged"] = True
    with pytest.raises(ValueError):
        MultiOutputPerceptron.from_dict(payload)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("weights_", np.array([[float("nan")], [0.0]])),
        ("bias_", np.zeros(3)),
        ("history_", [{"epoch": 1, "loss": 0.2, "accuracy": 0.5}]),
        ("updates_", True),
        ("converged_", 1),
    ],
)
def test_serialization_rejects_corrupted_live_state(attribute, value):
    model = MultiOutputPerceptron().fit([[0.0], [0.0]], [-1, 1], epochs=1)
    setattr(model, attribute, value)
    with pytest.raises(ValueError):
        model.to_dict()


@pytest.mark.parametrize("learning_rate", [0.05, 2.0])
def test_finite_but_overflowing_training_raises_value_error(learning_rate):
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="[Nn]on-finite|overflow|diverg"),
    ):
        MultiOutputPerceptron().fit(
            [[1e308], [-1e308]], [-1, 1], epochs=1, learning_rate=learning_rate
        )


@pytest.mark.parametrize("method", ["decision_function", "predict"])
def test_finite_but_overflowing_prediction_raises_value_error(method):
    model = MultiOutputPerceptron().fit([[-1.0], [1.0]], [-1, 1], epochs=1)
    model.weights_[:] = [[-1e308], [1e308]]
    model.bias_[:] = 0.0
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="[Nn]on-finite|overflow|diverg"),
    ):
        getattr(model, method)([[2.0]])


@pytest.mark.parametrize("option", ["batch_size", "tolerance"])
def test_fit_rejects_non_online_optimization_options(option):
    with pytest.raises(TypeError):
        MultiOutputPerceptron().fit([[-1.0], [1.0]], [-1, 1], **{option: 1})
