"""Stable float64 activations; samples occupy rows, classes the last axis."""

import numpy as np


def linear(value):
    return np.asarray(value, dtype=np.float64)


def relu(value):
    return np.maximum(0.0, value)


def leaky_relu(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value > 0.0, value, 0.02 * value)


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def tanh(value):
    return np.tanh(value)


def softmax(value):
    value = np.asarray(value, dtype=np.float64)
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


_ACTIVATIONS = {
    "linear": linear,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
}


def activation_name(value):
    """Canonicalize registered names or reference-style activation callables."""
    if value is None:
        return "linear"
    name = value if isinstance(value, str) else getattr(value, "__name__", None)
    if not isinstance(name, str):
        raise ValueError("Unsupported activation")  # noqa: TRY004 - consistent validation boundary
    name = {"none": "linear", "identity": "linear"}.get(name, name)
    if name not in _ACTIVATIONS:
        raise ValueError("Unsupported activation")
    return name


def activate(name, value):
    return _ACTIVATIONS[activation_name(name)](value)


def derivative(name, preactivation):
    """Elementwise derivative with respect to *preactivation*, not output."""
    name = activation_name(name)
    value = np.asarray(preactivation, dtype=np.float64)
    if name == "linear":
        return np.ones_like(value)
    if name == "relu":
        return (value > 0.0).astype(np.float64)
    if name == "leaky_relu":
        return np.where(value > 0.0, 1.0, 0.02)
    if name == "sigmoid":
        output = sigmoid(value)
        return output * (1.0 - output)
    if name == "tanh":
        return 1.0 - np.tanh(value) ** 2
    raise ValueError("Softmax requires the fused cross-entropy gradient")
