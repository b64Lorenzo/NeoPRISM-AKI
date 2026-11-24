"""Synthetic multinomial-regression tests."""

import numpy as np
import pytest

from src.models.logistic_regression import LogisticRegression


def test_multinomial_one_step_matches_mean_nll_gradient():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y = np.array([0, 1, 2, 3])
    model = LogisticRegression(seed=7)
    assert model.fit(X, y, epochs=1, learning_rate=0.2, batch_size=10) is model
    probabilities = np.full((4, 4), 0.25)
    expected_weights = -0.2 * (probabilities - np.eye(4)).T @ X / len(X)
    np.testing.assert_allclose(model.weights_, expected_weights)
    np.testing.assert_allclose(model.bias_, 0)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2, 3])
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
    assert model.n_features_in_ == 2
    assert model.history_[0]["epoch"] == 1
    assert model.history_[0]["accuracy"] == 1.0
    assert model.history_[0]["loss"] < np.log(4)


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_gradients_match_finite_differences_with_unregularized_bias(l2):
    X = np.array([[0.2, -0.3], [1.2, 0.7], [-0.8, 0.4]])
    y = np.array([0, 1, 2])
    model = LogisticRegression(l2=l2).fit(X, y, epochs=1)
    model.weights_[:] = [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.1]]
    model.bias_[:] = [0.4, -0.2, 0.1]

    def objective():
        logits = X @ model.weights_.T + model.bias_
        nll = np.mean(
            np.logaddexp.reduce(logits, axis=1) - logits[np.arange(len(y)), y]
        )
        return nll + l2 * np.sum(model.weights_**2) / 2

    actual_loss, grad_w, grad_b = model._loss_and_gradients(X, y)
    assert actual_loss == pytest.approx(objective())
    for parameter, gradient in ((model.weights_, grad_w), (model.bias_, grad_b)):
        numeric = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            old = parameter[index]
            parameter[index] = old + 1e-6
            positive = objective()
            parameter[index] = old - 1e-6
            negative = objective()
            parameter[index] = old
            numeric[index] = (positive - negative) / 2e-6
        np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-7)
    without_penalty = LogisticRegression().fit(X, y, epochs=1)
    without_penalty.weights_[:] = model.weights_
    without_penalty.bias_[:] = model.bias_
    _, plain_w, plain_b = without_penalty._loss_and_gradients(X, y)
    np.testing.assert_allclose(grad_w - plain_w, l2 * model.weights_, atol=1e-14)
    np.testing.assert_allclose(grad_b, plain_b)


@pytest.mark.parametrize(
    "X,y",
    [
        ([], []),
        ([[1, 2]], []),
        ([[1, 2]], [0]),
        ([[1, float("nan")], [2, 3]], [0, 1]),
        ([[1, float("inf")], [2, 3]], [0, 1]),
        ([[1, 2], [2, 3]], [0.2, 1.2]),
        ([[1, 2], [2, 3]], [1, "second"]),
        ([[1, 2], [2, 3]], [None, 1]),
        ([[1, 2], [2, 3]], [[0], [1]]),
        ([[1 + 2j], [2]], [0, 1]),
    ],
)
def test_rejects_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y, epochs=1)


@pytest.mark.parametrize(
    "arguments",
    [
        {"epochs": 0},
        {"epochs": True},
        {"epochs": 1.5},
        {"batch_size": 0},
        {"batch_size": 1.5},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"tolerance": -1},
        {"tolerance": float("inf")},
    ],
)
def test_rejects_invalid_fit_hyperparameters(arguments):
    with pytest.raises(ValueError):
        LogisticRegression().fit([[0], [1]], [-1, 1], **arguments)


@pytest.mark.parametrize(
    "arguments",
    [{"l2": -1}, {"l2": float("nan")}, {"l2": True}, {"seed": -1}, {"seed": True}],
)
def test_rejects_invalid_constructor_hyperparameters(arguments):
    with pytest.raises(ValueError):
        LogisticRegression(**arguments)


def test_prediction_requires_fitted_model_and_finite_feature_matrix():
    with pytest.raises(ValueError, match="fitted"):
        LogisticRegression().predict([[1]])
    model = LogisticRegression().fit([[0], [1]], [-1, 1], epochs=1)
    for X in ([1], [[1, 2]], [[float("nan")]]):
        with pytest.raises(ValueError):
            model.predict_proba(X)
    assert model.predict_proba(np.empty((0, 1))).shape == (0, 2)


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_tolerance_stops_only_when_enabled(kind):
    from src.models.neural_network import MultiLayerPerceptron

    def make_model():
        return (
            LogisticRegression() if kind == "logistic" else MultiLayerPerceptron([1, 1])
        )

    X, y = [[0], [0]], [-1, 1]
    stopped = make_model().fit(X, y, epochs=5, tolerance=1e-9)
    assert len(stopped.history_) == 2
    complete = make_model().fit(X, y, epochs=5, tolerance=0)
    assert len(complete.history_) == 5


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_remainder_batches_are_used_in_seeded_order(kind):
    from src.models.neural_network import MultiLayerPerceptron

    X = np.array([[-1.0], [0.2], [1.0]])
    y = np.array([-1, -1, 1])
    model = (
        LogisticRegression(seed=9)
        if kind == "logistic"
        else MultiLayerPerceptron([1, 2], init_methods=["zero"], seed=9)
    )
    weights, biases = np.zeros((2, 1)), np.zeros(2)
    order = np.random.default_rng(9).permutation(len(X))
    encoded = (y == 1).astype(int)
    for rows in (order[:2], order[2:]):
        logits = X[rows] @ weights.T + biases
        logs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        delta = (np.exp(logs) - np.eye(2)[encoded[rows]]) / len(rows)
        weights -= 0.2 * delta.T @ X[rows]
        biases -= 0.2 * delta.sum(axis=0)
    model.fit(X, y, epochs=1, learning_rate=0.2, batch_size=2)
    actual_w = model.weights_ if kind == "logistic" else model.weights_[0]
    actual_b = model.bias_ if kind == "logistic" else model.biases_[0]
    np.testing.assert_allclose(actual_w, weights)
    np.testing.assert_allclose(actual_b, biases)


@pytest.mark.parametrize(
    "y", [[-1, 1, -1, 1], [0, 1, 2, 3], ["low", "high", "low", "high"]]
)
def test_logistic_json_round_trip_preserves_class_order(y):
    import json

    X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    model = LogisticRegression(l2=0.2, seed=4).fit(X, y, epochs=3)
    payload = model.to_dict()
    restored = LogisticRegression.from_dict(
        json.loads(json.dumps(payload, allow_nan=False))
    )
    assert restored.to_dict() == payload
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(restored.classes_, model.classes_)
    restored.weights_[0, 0] += 1
    assert restored.to_dict() != payload


def test_unfitted_logistic_model_cannot_be_serialized():
    with pytest.raises(ValueError, match="fitted"):
        LogisticRegression().to_dict()


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", True),
        ("model_type", "unknown"),
        ("n_features_in", 2),
        ("n_features_in", True),
        ("classes", [-1, -1]),
        ("classes", [1, -1]),
        ("classes", [-1]),
        ("classes", [1, "positive"]),
        ("classes", [False, True]),
        ("classes", [-1.0, 1.0]),
        ("history", []),
        ("history", {}),
        ("history", [{"epoch": 0, "loss": 0.2, "accuracy": 0.5}]),
        ("history", [{"epoch": True, "loss": 0.2, "accuracy": 0.5}]),
        ("history", [{"epoch": 1, "loss": -0.1, "accuracy": 0.5}]),
        ("history", [{"epoch": 1, "loss": float("nan"), "accuracy": 0.5}]),
        ("history", [{"epoch": 1, "loss": 0.2, "accuracy": 1.1}]),
        ("history", [{"epoch": 1, "loss": 0.2, "accuracy": None}]),
        ("history", [{"epoch": 1, "loss": 0.2}]),
    ],
)
def test_malformed_model_metadata_is_rejected(kind, field, value):
    from src.models.neural_network import MultiLayerPerceptron

    model = LogisticRegression() if kind == "logistic" else MultiLayerPerceptron([1, 2])
    payload = model.fit([[0], [1]], [-1, 1], epochs=1).to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        type(model).from_dict(payload)


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "non_json",
        "bad_weights",
        "nan_weights",
        "bad_bias",
        "non_object",
    ],
)
def test_malformed_parameter_payloads_are_rejected(kind, case):
    from src.models.neural_network import MultiLayerPerceptron

    model = LogisticRegression() if kind == "logistic" else MultiLayerPerceptron([1, 2])
    payload = model.fit([[0], [1]], [-1, 1], epochs=1).to_dict()
    if case == "missing":
        del payload["weights"]
    elif case == "extra":
        payload["surprise"] = 1
    elif case == "non_json":
        payload["history"][0]["surprise"] = object()
    elif case == "bad_weights":
        payload["weights"] = []
    elif case == "nan_weights":
        payload["weights"] = (
            [[float("nan")], [0.0]] if kind == "logistic" else [[[float("nan")], [0.0]]]
        )
    elif case == "bad_bias":
        payload["bias" if kind == "logistic" else "biases"] = (
            [[0, 0]] if kind == "logistic" else [[0]]
        )
    elif case == "non_object":
        payload = []
    with pytest.raises(ValueError):
        type(model).from_dict(payload)


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_serialization_rejects_corrupted_live_model(kind):
    from src.models.neural_network import MultiLayerPerceptron

    model = LogisticRegression() if kind == "logistic" else MultiLayerPerceptron([1, 2])
    model.fit([[0], [1]], [-1, 1], epochs=1)
    model.history_[0]["accuracy"] = 2.0
    with pytest.raises(ValueError):
        model.to_dict()
    model.history_[0]["accuracy"] = 0.5
    model.weights_[0][0] = float("nan")
    with pytest.raises(ValueError):
        model.to_dict()


def test_logistic_extreme_logits_have_finite_unclipped_loss():
    X = np.array([[1.0], [-1.0]])
    model = LogisticRegression().fit(X, [-1, 1], epochs=1)
    model.weights_[:] = [[1000], [-1000]]
    model.bias_[:] = 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss, gradient_w, gradient_b = model._loss_and_gradients(X, np.array([1, 0]))
        probabilities = model.predict_proba(X)
    assert loss == pytest.approx(2000)
    assert all(
        np.isfinite(array).all() for array in (gradient_w, gradient_b, probabilities)
    )


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_fitting_is_fresh_deterministic_and_has_no_implicit_io(kind, monkeypatch):
    from src.models.neural_network import MultiLayerPerceptron

    X, y = [[-2], [-1], [0.5], [1], [2]], [-1, -1, 1, 1, 1]
    model = (
        LogisticRegression(seed=17)
        if kind == "logistic"
        else MultiLayerPerceptron([1, 3, 2], seed=17)
    )
    original_X, original_y = [row.copy() for row in X], y.copy()
    state = np.random.get_state()

    def forbid_io(*args, **kwargs):
        raise AssertionError("Numerical cores must not open files")

    with monkeypatch.context() as context:
        context.setattr("builtins.open", forbid_io)
        model.fit(X, y, epochs=10, learning_rate=0.1, batch_size=2)
        first = model.to_dict()
        model.fit(X, y, epochs=10, learning_rate=0.1, batch_size=2)
        assert model.to_dict() == first
        model.predict(X)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert X == original_X and y == original_y
