"""Small shared numerical helpers for the NumPy model cores."""
# ruff: noqa: N999, TRY004
# Preserve the reference filename and a consistent ValueError validation boundary.

from functools import wraps

import numpy as np


def numerically_checked(method):
    """Fail on arithmetic overflow instead of returning a silently invalid model."""

    @wraps(method)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return method(*args, **kwargs)
        except FloatingPointError as error:
            raise ValueError(
                "Non-finite numerical result; scale inputs or reduce the learning rate"
            ) from error

    return checked


def log_softmax(logits):
    """Compute log probabilities without logging underflowed probabilities."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def integer(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def real_number(value, name, *, positive=False):
    from numbers import Real

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return value


def fit_parameters(epochs, learning_rate, batch_size, tolerance):
    return (
        integer(epochs, "epochs"),
        real_number(learning_rate, "learning_rate", positive=True),
        integer(batch_size, "batch_size"),
        real_number(tolerance, "tolerance"),
    )


def float_array(value, name):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise ValueError(f"{name} must contain real numeric values")
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def features(X, n_features=None, *, allow_empty=False):
    X = float_array(X, "X")
    if X.ndim != 2 or X.shape[1] == 0 or (not allow_empty and X.shape[0] == 0):
        raise ValueError("X must be a nonempty two-dimensional feature matrix")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("X feature count does not match the model")
    return X


def labels(y, n_samples=None):
    try:
        array = np.asarray(y, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Labels must be a one-dimensional integer or string array"
        ) from exc
    if (
        array.ndim != 1
        or not array.size
        or (n_samples is not None and len(array) != n_samples)
    ):
        raise ValueError("Labels must be nonempty, one-dimensional, and aligned with X")
    values = array.tolist()
    ints = all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        for v in values
    )
    strings = all(isinstance(v, str) for v in values)
    if not (ints or strings):
        raise ValueError("Labels must be homogeneous integers or strings")
    try:
        # Fixed-width NumPy Unicode silently drops trailing NUL characters.
        return np.asarray(values, dtype=np.int64 if ints else object)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Integer labels must fit in int64") from exc


def encode_labels(y, n_samples):
    classes, encoded = np.unique(labels(y, n_samples), return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Classification requires at least two distinct classes")
    return classes, encoded


def artifact(payload, model_type, fields):
    """Validate a strict versioned JSON object and detach all caller containers."""
    import json

    if not isinstance(payload, dict) or set(payload) != set(fields):
        raise ValueError("Model artifact has missing or unknown fields")
    try:
        payload = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Model artifact must contain finite JSON-safe values") from exc
    if (
        integer(payload["version"], "version") != 1
        or payload["model_type"] != model_type
    ):
        raise ValueError("Unsupported model artifact type or version")
    return payload


def saved_classes(value, count=None):
    if not isinstance(value, list):
        raise ValueError("Saved classes must be a list")
    classes = labels(value)
    if len(classes) < 2 or not np.array_equal(classes, np.unique(classes)):
        raise ValueError("Saved classes must be sorted and unique")
    if count is not None and len(classes) != count:
        raise ValueError("Saved classes do not match output width")
    return classes


def saved_parameter(value, shape, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a numeric list")
    array = float_array(value, name)
    if array.shape != shape or any(
        isinstance(v, bool) for v in np.asarray(value, dtype=object).flat
    ):
        raise ValueError(f"{name} dimensions or values are invalid")
    return array.copy()


def saved_history(value, classification):
    if not isinstance(value, list) or not value:
        raise ValueError("Saved history must be a nonempty list")
    history = []
    for epoch, entry in enumerate(value, start=1):
        if not isinstance(entry, dict) or set(entry) != {"epoch", "loss", "accuracy"}:
            raise ValueError("History entries require epoch, loss and accuracy")
        if integer(entry["epoch"], "history epoch") != epoch:
            raise ValueError("History epochs must be consecutive from one")
        loss = real_number(entry["loss"], "history loss")
        accuracy = entry["accuracy"]
        if classification:
            accuracy = real_number(accuracy, "history accuracy")
            if accuracy > 1.0:
                raise ValueError("History accuracy must lie in [0, 1]")
        elif accuracy is not None:
            raise ValueError("Regression history accuracy must be null")
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
    return history
