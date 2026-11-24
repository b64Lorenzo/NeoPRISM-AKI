"""Seeded dense multilayer perceptron with explicit minibatch SGD."""
# ruff: noqa: N999, TRY004
# The reference filename and consistent ValueError boundary are public contracts.

import numpy as np

from .activation_func import activate, activation_name, derivative
from .nnUtils import (
    artifact,
    encode_labels,
    features,
    fit_parameters,
    float_array,
    integer,
    log_softmax,
    numerically_checked,
    saved_classes,
    saved_history,
    saved_parameter,
)


class MultiLayerPerceptron:
    """Row-sample arrays; inference returns post-activation network outputs."""

    def __init__(
        self,
        shape,
        activation_functions=None,
        init_methods=None,
        classification=True,
        loss=None,
        seed=2026,
    ):
        try:
            self.shape = [integer(width, "layer width") for width in shape]
        except TypeError as exc:
            raise ValueError(
                "shape must be a sequence of positive integer widths"
            ) from exc
        if len(self.shape) < 2:
            raise ValueError("shape must contain an input and output layer")
        if not isinstance(classification, bool):
            raise ValueError("classification must be boolean")
        self.classification = classification
        self.seed = integer(seed, "seed", minimum=0)
        output = "sigmoid" if self.shape[-1] == 1 else "softmax"
        defaults = ["relu"] * (len(self.shape) - 2) + [
            output if classification else "linear"
        ]
        self.activation_functions = [
            activation_name(a)
            for a in (
                defaults if activation_functions is None else activation_functions
            )
        ]
        if len(self.activation_functions) != len(self.shape) - 1:
            raise ValueError("Exactly one activation is required per non-input layer")
        if "softmax" in self.activation_functions[:-1]:
            raise ValueError("Softmax is only supported at the classification output")
        if self.activation_functions[-1] != defaults[-1]:
            raise ValueError(
                "Output activation does not match the task and output width"
            )
        self.init_methods = (
            list(init_methods)
            if init_methods is not None
            else [
                "he" if a in ("relu", "leaky_relu") else "xavier"
                for a in self.activation_functions
            ]
        )
        if len(self.init_methods) != len(self.shape) - 1 or any(
            method not in ("he", "xavier", "zero") for method in self.init_methods
        ):
            raise ValueError(
                "Exactly one registered initializer is required per non-input layer"
            )
        expected_loss = (
            "binary_cross_entropy"
            if classification and self.shape[-1] == 1
            else "cross_entropy"
            if classification
            else "mse"
        )
        aliases = {
            "MeanSquareError": "mse",
            "BinaryCrossEntropy": "binary_cross_entropy",
            "CategoricalCrossEntropy": "cross_entropy",
        }
        if loss is not None and not isinstance(loss, str):
            raise ValueError("loss must be a registered loss name")
        self.loss = expected_loss if loss is None else aliases.get(loss, loss)
        if self.loss != expected_loss:
            raise ValueError("Loss must match the task and output activation")
        self.history_ = []

    def _initialize(self, rng):
        self.weights_, self.biases_ = [], []
        for fan_in, fan_out, method in zip(
            self.shape[:-1], self.shape[1:], self.init_methods
        ):
            if method == "zero":
                weights = np.zeros((fan_out, fan_in), dtype=np.float64)
            else:
                scale = np.sqrt((2.0 if method == "he" else 1.0) / fan_in)
                weights = rng.standard_normal((fan_out, fan_in)) * scale
            self.weights_.append(weights)
            self.biases_.append(np.zeros(fan_out, dtype=np.float64))

    def _forward(self, X):
        activations, preactivations = [X], []
        for weights, biases, name in zip(
            self.weights_, self.biases_, self.activation_functions
        ):
            logits = activations[-1] @ weights.T + biases
            preactivations.append(logits)
            activations.append(activate(name, logits))
        return activations, preactivations

    def _loss_and_gradients(self, X, targets):
        activations, preactivations = self._forward(X)
        error = activations[-1] - targets
        if self.loss == "cross_entropy":
            loss = -np.sum(targets * log_softmax(preactivations[-1])) / len(X)
            delta = error / len(X)
        elif self.loss == "binary_cross_entropy":
            logits = preactivations[-1]
            loss = np.mean(np.logaddexp(0.0, logits) - targets * logits)
            delta = error / len(X)
        else:
            loss = 0.5 * np.mean(error**2)
            delta = error / targets.size
        weights_gradients, bias_gradients = [], []
        for index in range(len(self.weights_) - 1, -1, -1):
            if index != len(self.weights_) - 1 or not self.classification:
                delta = delta * derivative(
                    self.activation_functions[index], preactivations[index]
                )
            weights_gradients.append(delta.T @ activations[index])
            bias_gradients.append(delta.sum(axis=0))
            if index:
                delta = delta @ self.weights_[index]
        return float(loss), weights_gradients[::-1], bias_gradients[::-1]

    @numerically_checked
    def fit(
        self, X, y, *, epochs=1000, learning_rate=0.01, batch_size=32, tolerance=0.0
    ):
        epochs, learning_rate, batch_size, tolerance = fit_parameters(
            epochs, learning_rate, batch_size, tolerance
        )
        X = features(X, self.shape[0])
        if self.classification:
            classes, encoded_y = encode_labels(y, len(X))
            expected_classes = 2 if self.shape[-1] == 1 else self.shape[-1]
            if len(classes) != expected_classes:
                raise ValueError("Number of classes does not match the output width")
            self.classes_ = classes
            targets = (
                encoded_y[:, None]
                if self.shape[-1] == 1
                else np.eye(self.shape[-1])[encoded_y]
            )
        else:
            targets = float_array(y, "y")
            if targets.ndim == 1 and self.shape[-1] == 1:
                targets = targets[:, None]
            if targets.shape != (len(X), self.shape[-1]):
                raise ValueError(
                    "Regression targets must match samples and output width"
                )
            self.classes_ = None
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.seed)
        self._initialize(rng)
        self.history_ = []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                rows = order[start : start + batch_size]
                _, grad_w, grad_b = self._loss_and_gradients(X[rows], targets[rows])
                for index in range(len(self.weights_)):
                    self.weights_[index] -= learning_rate * grad_w[index]
                    self.biases_[index] -= learning_rate * grad_b[index]
            loss, _, _ = self._loss_and_gradients(X, targets)
            accuracy = (
                float(np.mean(self.predict(X) == self.classes_[encoded_y]))
                if self.classification
                else None
            )
            self.history_.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
            if tolerance > 0 and len(self.history_) > 1:
                improvement = self.history_[-2]["loss"] - loss
                if 0 <= improvement <= tolerance:
                    break
        return self

    @numerically_checked
    def inference(self, X):
        if not hasattr(self, "weights_"):
            raise ValueError("Model must be fitted before prediction")
        return self._forward(features(X, self.n_features_in_, allow_empty=True))[0][-1]

    def predict(self, X):
        if self.classification:
            probabilities = self.predict_proba(X)
            return self.classes_[np.argmax(probabilities, axis=1)]
        result = self.inference(X)
        return result[:, 0] if self.shape[-1] == 1 else result

    def predict_proba(self, X):
        if not self.classification:
            raise ValueError("Regression does not expose class probabilities")
        result = self.inference(X)
        return (
            np.column_stack((1.0 - result[:, 0], result[:, 0]))
            if self.shape[-1] == 1
            else result
        )

    def to_dict(self):
        if not hasattr(self, "weights_"):
            raise ValueError("Model must be fitted before serialization")
        payload = {
            "version": 1,
            "model_type": "mlp",
            "shape": self.shape.copy(),
            "activation_functions": self.activation_functions.copy(),
            "init_methods": self.init_methods.copy(),
            "classification": self.classification,
            "loss": self.loss,
            "seed": self.seed,
            "n_features_in": self.n_features_in_,
            "classes": self.classes_.tolist() if self.classification else None,
            "weights": [weights.tolist() for weights in self.weights_],
            "biases": [bias.tolist() for bias in self.biases_],
            "history": [dict(entry) for entry in self.history_],
        }
        self.from_dict(payload)
        return payload

    @classmethod
    def from_dict(cls, payload):
        payload = artifact(
            payload,
            "mlp",
            {
                "version",
                "model_type",
                "shape",
                "activation_functions",
                "init_methods",
                "classification",
                "loss",
                "seed",
                "n_features_in",
                "classes",
                "weights",
                "biases",
                "history",
            },
        )
        if any(
            not isinstance(payload[key], list)
            for key in (
                "shape",
                "activation_functions",
                "init_methods",
                "weights",
                "biases",
            )
        ):
            raise ValueError("Network configuration and parameters must be lists")
        model = cls(
            payload["shape"],
            payload["activation_functions"],
            payload["init_methods"],
            payload["classification"],
            payload["loss"],
            payload["seed"],
        )
        model.n_features_in_ = integer(payload["n_features_in"], "n_features_in")
        if model.n_features_in_ != model.shape[0]:
            raise ValueError("Saved feature count does not match network input width")
        count = 2 if model.shape[-1] == 1 else model.shape[-1]
        model.classes_ = (
            saved_classes(payload["classes"], count) if model.classification else None
        )
        if not model.classification and payload["classes"] is not None:
            raise ValueError("Regression must not contain class labels")
        if (
            len(payload["weights"]) != len(model.shape) - 1
            or len(payload["biases"]) != len(model.shape) - 1
        ):
            raise ValueError("Saved parameters do not match network layer count")
        model.weights_, model.biases_ = [], []
        for index, (fan_in, fan_out) in enumerate(
            zip(model.shape[:-1], model.shape[1:])
        ):
            model.weights_.append(
                saved_parameter(payload["weights"][index], (fan_out, fan_in), "weights")
            )
            model.biases_.append(
                saved_parameter(payload["biases"][index], (fan_out,), "biases")
            )
        model.history_ = saved_history(payload["history"], model.classification)
        return model
