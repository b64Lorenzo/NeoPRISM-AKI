"""Minibatch multinomial logistic regression with a mean-NLL objective."""

import numpy as np

from ..neural_network.nnUtils import (
    artifact,
    encode_labels,
    features,
    fit_parameters,
    integer,
    log_softmax,
    numerically_checked,
    real_number,
    saved_classes,
    saved_history,
    saved_parameter,
)


class LogisticRegression:
    """Full K-class softmax, including K=2; labels remain externally visible."""

    def __init__(self, l2=0.0, seed=2026):
        self.l2 = real_number(l2, "l2")
        self.seed = integer(seed, "seed", minimum=0)
        self.history_ = []

    def _loss_and_gradients(self, X, encoded_y):
        logs = log_softmax(X @ self.weights_.T + self.bias_)
        loss = -logs[np.arange(len(X)), encoded_y].mean()
        loss += self.l2 * np.sum(self.weights_**2) / 2.0
        delta = np.exp(logs)
        delta[np.arange(len(X)), encoded_y] -= 1.0
        delta /= len(X)
        return float(loss), delta.T @ X + self.l2 * self.weights_, delta.sum(axis=0)

    @numerically_checked
    def fit(
        self, X, y, *, epochs=1000, learning_rate=0.01, batch_size=32, tolerance=0.0
    ):
        epochs, learning_rate, batch_size, tolerance = fit_parameters(
            epochs, learning_rate, batch_size, tolerance
        )
        X = features(X)
        self.classes_, encoded_y = encode_labels(y, len(X))
        self.n_features_in_ = X.shape[1]
        self.weights_ = np.zeros(
            (len(self.classes_), self.n_features_in_), dtype=np.float64
        )
        self.bias_ = np.zeros(len(self.classes_), dtype=np.float64)
        self.history_ = []
        rng = np.random.default_rng(self.seed)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                rows = order[start : start + batch_size]
                _, weights_gradient, bias_gradient = self._loss_and_gradients(
                    X[rows], encoded_y[rows]
                )
                self.weights_ -= learning_rate * weights_gradient
                self.bias_ -= learning_rate * bias_gradient
            loss, _, _ = self._loss_and_gradients(X, encoded_y)
            accuracy = float(np.mean(self.predict(X) == self.classes_[encoded_y]))
            self.history_.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
            if tolerance > 0 and len(self.history_) > 1:
                improvement = self.history_[-2]["loss"] - loss
                if 0 <= improvement <= tolerance:
                    break
        return self

    @numerically_checked
    def predict_proba(self, X):
        if not hasattr(self, "weights_"):
            raise ValueError("Model must be fitted before prediction")
        X = features(X, self.n_features_in_, allow_empty=True)
        return np.exp(log_softmax(X @ self.weights_.T + self.bias_))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def to_dict(self):
        if not hasattr(self, "weights_"):
            raise ValueError("Model must be fitted before serialization")
        payload = {
            "version": 1,
            "model_type": "logistic_regression",
            "l2": self.l2,
            "seed": self.seed,
            "n_features_in": self.n_features_in_,
            "classes": self.classes_.tolist(),
            "weights": self.weights_.tolist(),
            "bias": self.bias_.tolist(),
            "history": [dict(entry) for entry in self.history_],
        }
        self.from_dict(payload)
        return payload

    @classmethod
    def from_dict(cls, payload):
        payload = artifact(
            payload,
            "logistic_regression",
            {
                "version",
                "model_type",
                "l2",
                "seed",
                "n_features_in",
                "classes",
                "weights",
                "bias",
                "history",
            },
        )
        model = cls(l2=payload["l2"], seed=payload["seed"])
        model.n_features_in_ = integer(payload["n_features_in"], "n_features_in")
        model.classes_ = saved_classes(payload["classes"])
        count = len(model.classes_)
        model.weights_ = saved_parameter(
            payload["weights"], (count, model.n_features_in_), "weights"
        )
        model.bias_ = saved_parameter(payload["bias"], (count,), "bias")
        model.history_ = saved_history(payload["history"], True)
        return model
