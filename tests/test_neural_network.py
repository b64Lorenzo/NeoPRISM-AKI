"""Synthetic numerical tests; no clinical data or filesystem model I/O."""

import numpy as np
import pytest

from src.models.neural_network import activation_func as af


def test_stable_activations_extreme_logits():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        np.testing.assert_allclose(
            af.sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0, 0.5, 1]
        )
        values = np.array([[1000.0, 999.0, -1000.0], [-1000.0, 0.0, 1000.0]])
        probabilities = af.softmax(values)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(af.softmax(values + 10000), probabilities)
    assert probabilities[0, 0] > probabilities[0, 1]


@pytest.mark.parametrize("name", ["linear", "relu", "leaky_relu", "sigmoid", "tanh"])
def test_activation_derivatives_match_finite_differences(name):
    values = np.array([-1.3, -0.4, 0.3, 1.7])
    epsilon = 1e-6
    numeric = (
        af.activate(name, values + epsilon) - af.activate(name, values - epsilon)
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        af.derivative(name, values), numeric, rtol=1e-5, atol=1e-7
    )


def test_activation_reference_callables_have_canonical_names():
    for name in ("relu", "leaky_relu", "sigmoid", "softmax"):
        assert af.activation_name(getattr(af, name)) == name
    assert af.activation_name(None) == "linear"
    assert af.activation_name("none") == "linear"
    with pytest.raises(ValueError):
        af.activation_name(lambda x: x)


@pytest.mark.parametrize("outputs", [1, 2])
def test_regression_one_step_matches_mean_over_samples_and_outputs(outputs):
    from src.models.neural_network import MultiLayerPerceptron

    X = np.array([[1.0, 0.5], [-0.5, 1.0], [0.2, -0.3]])
    y = np.array([[0.5, -0.3], [-0.2, 0.8], [0.4, 0.2]])[:, :outputs]
    supplied_y = y[:, 0] if outputs == 1 else y
    model = MultiLayerPerceptron(
        [2, outputs], init_methods=["zero"], classification=False
    )
    assert model.fit(X, supplied_y, epochs=1, learning_rate=0.2, batch_size=10) is model
    np.testing.assert_allclose(model.weights_[0], 0.2 * y.T @ X / (len(X) * outputs))
    np.testing.assert_allclose(model.biases_[0], 0.2 * y.mean(axis=0) / outputs)
    result = model.inference(X)
    assert result.shape == (3, outputs)
    assert model.predict(X).shape == ((3,) if outputs == 1 else (3, outputs))
    assert model.history_ == [
        {
            "epoch": 1,
            "loss": pytest.approx(0.5 * np.mean((result - y) ** 2)),
            "accuracy": None,
        }
    ]
    with pytest.raises(ValueError):
        model.predict_proba(X)


@pytest.mark.parametrize("outputs,labels", [(1, [-1, 1, -1, 1]), (4, [0, 1, 2, 3])])
def test_classification_one_step_uses_fused_logit_gradient(outputs, labels):
    from src.models.neural_network import MultiLayerPerceptron

    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y = np.asarray(labels)
    classes, encoded = np.unique(y, return_inverse=True)
    targets = encoded[:, None] if outputs == 1 else np.eye(outputs)[encoded]
    initial = np.full_like(targets, 0.5 if outputs == 1 else 0.25, dtype=float)
    model = MultiLayerPerceptron([2, outputs], init_methods=["zero"])
    model.fit(X, y, epochs=1, learning_rate=0.2, batch_size=10)
    np.testing.assert_allclose(
        model.weights_[0], -0.2 * (initial - targets).T @ X / len(X)
    )
    np.testing.assert_allclose(
        model.biases_[0], -0.2 * (initial - targets).mean(axis=0)
    )
    np.testing.assert_array_equal(model.classes_, classes)
    raw = model.inference(X)
    probabilities = model.predict_proba(X)
    assert raw.shape == (4, outputs)
    assert probabilities.shape == (4, len(classes))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    if outputs == 1:
        np.testing.assert_allclose(probabilities[:, 1], raw[:, 0])
    else:
        np.testing.assert_allclose(probabilities, raw)
    expected = classes[np.argmax(probabilities, axis=1)]
    np.testing.assert_array_equal(model.predict(X), expected)
    assert model.history_[0]["accuracy"] == np.mean(expected == y)


@pytest.mark.parametrize(
    "shape,arguments",
    [
        ([2], {}),
        ([2, 0, 1], {}),
        ([2, 2.5, 1], {}),
        ([2, True], {}),
        ([2, 1], {"classification": "yes"}),
        ([2, 3, 1], {"activation_functions": ["sigmoid"]}),
        ([2, 3, 1], {"activation_functions": ["softmax", "sigmoid"]}),
        ([2, 1], {"activation_functions": ["relu"]}),
        ([2, 2], {"loss": "mse"}),
        ([2, 2], {"loss": "binary_cross_entropy"}),
        ([2, 1], {"classification": False, "activation_functions": ["sigmoid"]}),
        ([2, 1], {"init_methods": []}),
        ([2, 1], {"init_methods": ["bogus"]}),
        ([2, 1], {"seed": -1}),
    ],
)
def test_invalid_network_configuration_is_rejected(shape, arguments):
    from src.models.neural_network import MultiLayerPerceptron

    with pytest.raises(ValueError):
        MultiLayerPerceptron(shape, **arguments)


@pytest.mark.parametrize(
    "outputs,classification,y",
    [
        (1, True, [1, 1]),
        (1, True, [-1, 0, 1]),
        (4, True, [0, 1]),
        (1, True, [1, "one"]),
        (1, True, [0.5, 1.5]),
        (1, False, [[1, 2], [3, 4]]),
        (2, False, [1, 2]),
        (1, False, [1, float("nan")]),
    ],
)
def test_invalid_targets_are_rejected(outputs, classification, y):
    from src.models.neural_network import MultiLayerPerceptron

    with pytest.raises(ValueError):
        MultiLayerPerceptron([1, outputs], classification=classification).fit(
            [[0], [1]], y, epochs=1
        )


def test_network_fit_and_prediction_validate_boundaries():
    from src.models.neural_network import MultiLayerPerceptron

    model = MultiLayerPerceptron([1, 1])
    with pytest.raises(ValueError, match="fitted"):
        model.inference([[0]])
    for arguments in (
        {"epochs": 0},
        {"batch_size": True},
        {"learning_rate": -1},
        {"tolerance": float("nan")},
    ):
        with pytest.raises(ValueError):
            model.fit([[0], [1]], [-1, 1], **arguments)
    for X in ([], [[0, 1], [2, 3]], [[0], [float("nan")]]):
        with pytest.raises(ValueError):
            model.fit(X, [-1, 1], epochs=1)
    model.fit([[0], [1]], [-1, 1], epochs=1)
    for X in ([0], [[0, 1]], [[float("inf")]]):
        with pytest.raises(ValueError):
            model.predict(X)
    assert model.inference(np.empty((0, 1))).shape == (0, 1)


def test_reference_loss_names_and_callable_activations_are_canonicalized():
    from src.models.neural_network import MultiLayerPerceptron

    model = MultiLayerPerceptron(
        [1, 2, 1], [af.relu, af.sigmoid], loss="BinaryCrossEntropy"
    )
    assert model.activation_functions == ["relu", "sigmoid"]
    assert model.loss == "binary_cross_entropy"
    regressor = MultiLayerPerceptron(
        [1, 1], [None], classification=False, loss="MeanSquareError"
    )
    assert regressor.loss == "mse"


@pytest.mark.parametrize("task", ["binary", "stage", "regression"])
def test_neural_serialization_round_trip_is_json_safe(task):
    import json

    from src.models.neural_network import MultiLayerPerceptron

    X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    outputs = 4 if task == "stage" else 1
    model = MultiLayerPerceptron([2, 3, outputs], classification=task != "regression")
    y = (
        [0, 1, 2, 3]
        if task == "stage"
        else ["negative", "positive", "positive", "negative"]
        if task == "binary"
        else [0.2, -0.3, 0.5, 0.7]
    )
    model.fit(X, y, epochs=3)
    payload = model.to_dict()
    restored = MultiLayerPerceptron.from_dict(
        json.loads(json.dumps(payload, allow_nan=False))
    )
    assert restored.to_dict() == payload
    np.testing.assert_allclose(restored.inference(X), model.inference(X))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.history_ == model.history_
    if task != "regression":
        np.testing.assert_array_equal(restored.classes_, model.classes_)
    restored.weights_[0][0, 0] += 1
    assert restored.to_dict() != payload


def test_unfitted_neural_network_cannot_be_serialized():
    from src.models.neural_network import MultiLayerPerceptron

    with pytest.raises(ValueError, match="fitted"):
        MultiLayerPerceptron([1, 1]).to_dict()


@pytest.mark.parametrize("hidden", ["linear", "relu", "leaky_relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_deep_network_gradients_match_finite_differences(hidden, task):
    from src.models.neural_network import MultiLayerPerceptron

    X = np.array([[0.2, -0.3], [1.2, 0.7], [-0.8, 0.4], [0.5, -0.2]])
    outputs = 1 if task == "binary" else 3 if task == "multiclass" else 2
    final = (
        "sigmoid"
        if task == "binary"
        else "softmax"
        if task == "multiclass"
        else "linear"
    )
    y = (
        [-1, 1, -1, 1]
        if task == "binary"
        else [0, 1, 2, 0]
        if task == "multiclass"
        else [[0.2, -0.1], [0.4, 0.7], [-0.3, 0.1], [0.1, 0.2]]
    )
    targets = (
        np.array([[0], [1], [0], [1]])
        if task == "binary"
        else np.eye(3)[y]
        if task == "multiclass"
        else np.asarray(y)
    )
    model = MultiLayerPerceptron(
        [2, 3, 2, outputs], [hidden, "tanh", final], classification=task != "regression"
    )
    model.fit(X, y, epochs=1)
    model.weights_[0][:] = [[0.2, -0.1], [-0.3, 0.4], [0.1, 0.25]]
    model.biases_[0][:] = [0.4, -0.2, 0.3]

    def objective():
        value = X
        for weights, bias, activation in zip(
            model.weights_, model.biases_, model.activation_functions
        ):
            logits = value @ weights.T + bias
            value = af.activate(activation, logits)
        if task == "binary":
            return np.mean(np.logaddexp(0, logits) - targets * logits)
        if task == "multiclass":
            return np.mean(
                np.logaddexp.reduce(logits, axis=1) - np.sum(targets * logits, axis=1)
            )
        return 0.5 * np.mean((value - targets) ** 2)

    loss, gradients_w, gradients_b = model._loss_and_gradients(X, targets)
    assert loss == pytest.approx(objective())
    for parameter, gradient in list(zip(model.weights_, gradients_w)) + list(
        zip(model.biases_, gradients_b)
    ):
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


@pytest.mark.parametrize("outputs", [1, 2])
def test_default_relu_network_learns_xor(outputs):
    from src.models.neural_network import MultiLayerPerceptron

    X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    y = [-1, 1, 1, -1]
    model = MultiLayerPerceptron([2, 8, outputs]).fit(
        X, y, epochs=1500, learning_rate=0.15, batch_size=4
    )
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.history_[-1]["loss"] < 0.03
    assert model.history_[-1]["accuracy"] == 1


@pytest.mark.parametrize("outputs", [1, 2])
def test_affine_single_and_multioutput_regression_converges(outputs):
    from src.models.neural_network import MultiLayerPerceptron

    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    y = (X @ np.array([[0.3, -0.2], [0.8, 0.4]]) + [0.1, -0.3])[:, :outputs]
    model = MultiLayerPerceptron([2, outputs], classification=False).fit(
        X, y, epochs=500, learning_rate=0.1, batch_size=4
    )
    np.testing.assert_allclose(model.inference(X), y, atol=1e-6)
    assert model.history_[-1]["loss"] < model.history_[0]["loss"]


@pytest.mark.parametrize("outputs", [1, 2])
def test_classification_loss_is_finite_and_not_probability_clipped(outputs):
    from src.models.neural_network import MultiLayerPerceptron

    X = np.array([[1.0], [-1.0]])
    model = MultiLayerPerceptron([1, outputs]).fit(X, [-1, 1], epochs=1)
    model.weights_[0][:] = [[1000]] if outputs == 1 else [[1000], [-1000]]
    model.biases_[0][:] = 0
    targets = np.array([[0], [1]]) if outputs == 1 else np.array([[0, 1], [1, 0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        loss, weights, biases = model._loss_and_gradients(X, targets)
        probabilities = model.predict_proba(X)
    assert loss == pytest.approx(1000 if outputs == 1 else 2000)
    assert all(np.isfinite(array).all() for array in weights + biases)
    assert np.isfinite(probabilities).all()


def test_worsening_loss_does_not_trigger_tolerance_stop():
    from src.models.neural_network import MultiLayerPerceptron

    model = MultiLayerPerceptron([1, 1], init_methods=["zero"], classification=False)
    model.fit([[1], [2]], [1, 2], epochs=4, learning_rate=1, tolerance=1e9)
    assert len(model.history_) == 4
    assert model.history_[-1]["loss"] > model.history_[0]["loss"]
