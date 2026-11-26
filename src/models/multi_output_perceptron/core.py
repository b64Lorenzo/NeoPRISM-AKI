"""Sequential multiclass perceptron with one raw score per class."""

import numpy as np

from ..neural_network.nnUtils import (
    artifact,
    encode_labels,
    features,
    integer,
    numerically_checked,
    real_number,
    saved_classes,
    saved_history,
    saved_parameter,
)


class MultiOutputPerceptron:
    """Single-label linear classifier; scores are not class probabilities.

    Samples occupy rows. Ties select the first sorted class. History loss is
    post-epoch misclassification rate, not a cross-entropy objective. updates_
    counts mistaken samples; converged_ requires an epoch without online mistakes.
    """

    def __init__(self, seed=2026):
        self.seed = integer(seed, "seed", minimum=0)
        self.history_ = []

    @numerically_checked
    def fit(self, X, y, *, epochs=1000, learning_rate=0.05):
        """Fit fresh zero weights using seeded, sequential mistake updates."""
        epochs = integer(epochs, "epochs")
        learning_rate = real_number(learning_rate, "learning_rate", positive=True)
        X = features(X)
        self.classes_, encoded_y = encode_labels(y, len(X))
        self.n_features_in_ = X.shape[1]
        self.weights_ = np.zeros(
            (len(self.classes_), self.n_features_in_), dtype=np.float64
        )
        self.bias_ = np.zeros(len(self.classes_), dtype=np.float64)
        self.history_ = []
        self.updates_ = 0
        self.converged_ = False
        rng = np.random.default_rng(self.seed)
        for epoch in range(1, epochs + 1):
            online_mistakes = 0
            for row in rng.permutation(len(X)):
                predicted = np.argmax(self.weights_ @ X[row] + self.bias_)
                actual = encoded_y[row]
                if predicted != actual:
                    self.weights_[actual] += learning_rate * X[row]
                    self.weights_[predicted] -= learning_rate * X[row]
                    self.bias_[actual] += learning_rate
                    self.bias_[predicted] -= learning_rate
                    self.updates_ += 1
                    online_mistakes += 1
            accuracy = float(np.mean(self.predict(X) == self.classes_[encoded_y]))
            self.history_.append(
                {"epoch": epoch, "loss": 1.0 - accuracy, "accuracy": accuracy}
            )
            if online_mistakes == 0:
                self.converged_ = True
                break
        return self

    @numerically_checked
    def decision_function(self, X):
        """Return raw scores in classes_ order with shape (samples, classes)."""
        if not hasattr(self, "weights_"):
            raise ValueError("Model must be fitted before prediction")
        X = features(X, self.n_features_in_, allow_empty=True)
        return X @ self.weights_.T + self.bias_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def to_dict(self):
        """Return a detached version-one JSON-safe model snapshot."""
        if not hasattr(self, "weights_"):
            raise ValueError("Model must be fitted before serialization")
        payload = {
            "version": 1,
            "model_type": "multi_output_perceptron",
            "seed": self.seed,
            "n_features_in": self.n_features_in_,
            "classes": self.classes_.tolist(),
            "weights": self.weights_.tolist(),
            "bias": self.bias_.tolist(),
            "history": [dict(entry) for entry in self.history_],
            "updates": self.updates_,
            "converged": self.converged_,
        }
        self.from_dict(payload)
        return payload

    @classmethod
    def from_dict(cls, payload):
        """Restore parameters and diagnostics without sharing caller containers."""
        payload = artifact(
            payload,
            "multi_output_perceptron",
            {
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
            },
        )
        model = cls(seed=payload["seed"])
        model.n_features_in_ = integer(payload["n_features_in"], "n_features_in")
        model.classes_ = saved_classes(payload["classes"])
        count = len(model.classes_)
        model.weights_ = saved_parameter(
            payload["weights"], (count, model.n_features_in_), "weights"
        )
        model.bias_ = saved_parameter(payload["bias"], (count,), "bias")
        model.history_ = saved_history(payload["history"], True)
        if any(
            entry["loss"] > 1.0
            or not np.isclose(
                entry["loss"], 1.0 - entry["accuracy"], rtol=0.0, atol=1e-12
            )
            for entry in model.history_
        ):
            raise ValueError("History loss must be the misclassification rate")
        model.updates_ = integer(payload["updates"], "updates", minimum=0)
        if not isinstance(payload["converged"], bool):
            # Keep the numerical cores' consistent ValueError validation boundary.
            raise ValueError("Saved convergence flag must be boolean")  # noqa: TRY004
        model.converged_ = payload["converged"]
        if model.updates_ < len(model.history_) - int(model.converged_):
            raise ValueError("Saved update count is inconsistent with epoch history")
        if model.converged_ and (
            len(model.history_) < 2 or model.history_[-1]["accuracy"] != 1.0
        ):
            raise ValueError("Online convergence requires a final mistake-free epoch")
        return model
