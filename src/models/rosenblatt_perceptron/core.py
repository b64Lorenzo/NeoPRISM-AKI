"""Course Rosenblatt algorithm with a patient-workflow adapter and diagnostics.

Reference: nnci-course-python/A1assignment/perceptron_rosenblatt.py.
The canonical loop preserves fixed order, division by the effective dimension,
strict positive-margin convergence, and per-pattern embedding strengths.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass
class Perceptron:
    weights: list[float]
    bias: float = 0.0
    threshold: float = 0.0

    def score(self, x):
        if len(x) != len(self.weights) or not all(isfinite(float(v)) for v in x):
            raise ValueError("Invalid feature vector")
        return sum(w * float(v) for w, v in zip(self.weights, x)) + self.bias

    def predict(self, x):
        return 1 if self.score(x) >= 0 else -1


def _train_rosenblatt(examples, max_epochs, threshold, clamped, history=None):
    if max_epochs < 1 or not isfinite(threshold):
        raise ValueError("Maximum epochs must be positive and threshold must be finite")
    if not examples or not examples[0][0]:
        raise ValueError("Training data must not be empty")

    dimensions = len(examples[0][0])
    for inputs, label in examples:
        if len(inputs) != dimensions or label not in (-1, 1):
            raise ValueError(
                "Inputs must have equal dimensions and labels must be -1 or +1"
            )
        if any(not isfinite(value) for value in inputs):
            raise ValueError("Inputs must be finite")

    if clamped:
        examples = [([*inputs, 1.0], label) for inputs, label in examples]
        dimensions += 1

    weights = [0.0] * dimensions
    strengths = [0] * len(examples)
    updates = 0
    for epoch in range(max_epochs):
        updates_before = updates
        for index, (inputs, label) in enumerate(examples):
            potential = label * sum(
                weight * value for weight, value in zip(weights, inputs)
            )
            if potential <= threshold:
                for feature in range(dimensions):
                    weights[feature] += label * inputs[feature] / dimensions
                strengths[index] += 1
                updates += 1

        margins = [
            label * sum(weight * value for weight, value in zip(weights, inputs))
            for inputs, label in examples
        ]
        converged = all(margin > 0.0 for margin in margins)
        if history is not None:
            history.append(
                {
                    "epoch": epoch + 1,
                    "updates": updates,
                    "updates_epoch": updates - updates_before,
                    "violations": sum(margin <= threshold for margin in margins),
                    "nonpositive_margins": sum(margin <= 0.0 for margin in margins),
                    "training_accuracy": sum(margin > 0.0 for margin in margins)
                    / len(margins),
                    "mean_signed_margin": sum(margins) / len(margins),
                    "perceptron_loss": sum(max(0.0, -margin) for margin in margins)
                    / len(margins),
                }
            )
        if converged:
            return weights, epoch + 1, updates, True, strengths
    return weights, max_epochs, updates, False, strengths


def train_rosenblatt(examples, max_epochs=100, threshold=0.0, clamped=False):
    """Return (weights, epochs, updates, converged, strengths), as in the course."""
    return _train_rosenblatt(examples, max_epochs, threshold, clamped)


def train(X, y, *, epochs=100, threshold=0.0, clamped=False):
    """Run the same algorithm, exposing model state and epoch diagnostics."""
    if len(X) != len(y):
        raise ValueError("Training requires aligned examples")
    history = []
    weights, epochs_run, updates, converged, strengths = _train_rosenblatt(
        list(zip(X, y)),
        epochs,
        threshold,
        clamped,
        history,
    )
    dimensions = len(weights)
    model = Perceptron(
        weights[:-1] if clamped else weights, weights[-1] if clamped else 0.0, threshold
    )
    return model, {
        "epochs_run": epochs_run,
        "updates": updates,
        "converged": converged,
        "strengths": strengths,
        "history": history,
        "max_epochs": epochs,
        "learning_rate": 1.0 / dimensions,
        "threshold": threshold,
        "clamped": clamped,
        "effective_dimensions": dimensions,
        "training_order": "fixed",
    }
