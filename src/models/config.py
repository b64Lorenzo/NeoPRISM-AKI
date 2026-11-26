"""Immutable classifier INI settings; importing this module loads no backend."""

from __future__ import annotations

import configparser
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.models.contracts import HIDDEN_ACTIVATIONS


@dataclass(frozen=True, slots=True)
class RaySettings:
    num_cpus: int = 1
    cpus_per_task: float = 1.0
    timeout_seconds: float = 300.0
    temp_directory: Path | None = None


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    epochs: int = 1000
    learning_rate: float = 0.05
    batch_size: int = 32
    tolerance: float = 0.0


@dataclass(frozen=True, slots=True)
class OnlineSettings:
    epochs: int = 1000
    learning_rate: float = 0.05


@dataclass(frozen=True, slots=True)
class MLPSettings:
    training: TrainingSettings = field(default_factory=TrainingSettings)
    hidden_layers: tuple[int, ...] = (16,)
    hidden_activations: tuple[str, ...] = ("relu",)
    init_methods: tuple[str, ...] | None = None
    output_activation: str = "softmax"
    loss: str = "cross_entropy"
    model_type: Literal["multilayer_perceptron"] = field(
        default="multilayer_perceptron", init=False
    )


@dataclass(frozen=True, slots=True)
class MultiOutputSettings:
    training: OnlineSettings = field(default_factory=OnlineSettings)
    output_activation: str = "linear"
    loss: str = "mistake_rate"
    model_type: Literal["multi_output_perceptron"] = field(
        default="multi_output_perceptron", init=False
    )


@dataclass(frozen=True, slots=True)
class LogisticSettings:
    training: TrainingSettings = field(default_factory=TrainingSettings)
    l2: float = 0.0
    output_activation: str = "softmax"
    loss: str = "cross_entropy"
    model_type: Literal["logistic_regression"] = field(
        default="logistic_regression", init=False
    )


@dataclass(frozen=True, slots=True)
class ClassifierSettings:
    ray: RaySettings
    model: MLPSettings | MultiOutputSettings | LogisticSettings


class _ClassifierParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _integer(value: str, name: str) -> int:
    if not re.fullmatch(r"[+-]?[0-9]+", value) or int(value) < 1:
        raise ValueError(f"{name} must be a positive decimal integer")
    return int(value)


def _number(value: str, name: str, *, positive: bool = False) -> float:
    if not re.fullmatch(
        r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value
    ):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {bound} number")
    return number


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.split(",")) if value else ()


def _online(raw: Mapping[str, str]) -> OnlineSettings:
    return OnlineSettings(
        epochs=_integer(raw.get("epochs", "1000"), "epochs"),
        learning_rate=_number(
            raw.get("learning_rate", "0.05"), "learning_rate", positive=True
        ),
    )


def _gradient(raw: Mapping[str, str]) -> TrainingSettings:
    online = _online(raw)
    return TrainingSettings(
        epochs=online.epochs,
        learning_rate=online.learning_rate,
        batch_size=_integer(raw.get("batch_size", "32"), "batch_size"),
        tolerance=_number(raw.get("tolerance", "0"), "tolerance"),
    )


_ONLINE_KEYS = {"model_type", "epochs", "learning_rate", "output_activation", "loss"}
_GRADIENT_KEYS = _ONLINE_KEYS | {"batch_size", "tolerance"}
_MODEL_KEYS = {
    "multilayer_perceptron": _GRADIENT_KEYS
    | {"hidden_layers", "hidden_activations", "init_methods"},
    "multi_output_perceptron": _ONLINE_KEYS,
    "logistic_regression": _GRADIENT_KEYS | {"l2"},
}


def _check_keys(raw: Mapping[str, str], allowed: set[str]) -> None:
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown classifier INI key: {min(unknown)}")


def _model_settings(
    raw: Mapping[str, str],
) -> MLPSettings | MultiOutputSettings | LogisticSettings:
    model_type = raw.get("model_type", "")
    if model_type not in _MODEL_KEYS:
        raise ValueError("Every classifier profile requires a supported model_type")
    _check_keys(raw, _MODEL_KEYS[model_type])
    output, loss = (
        ("linear", "mistake_rate")
        if model_type == "multi_output_perceptron"
        else ("softmax", "cross_entropy")
    )
    for key, expected in (("output_activation", output), ("loss", loss)):
        if raw.get(key, expected) != expected:
            raise ValueError(f"{key} must be {expected} for {model_type}")
    if model_type == "multi_output_perceptron":
        return MultiOutputSettings(training=_online(raw))
    training = _gradient(raw)
    if model_type == "logistic_regression":
        return LogisticSettings(training=training, l2=_number(raw.get("l2", "0"), "l2"))
    widths = tuple(
        _integer(token, "hidden_layers")
        for token in _tokens(raw.get("hidden_layers", "16"))
    )
    activations = (
        _tokens(raw["hidden_activations"])
        if "hidden_activations" in raw
        else ("relu",) * len(widths)
    )
    if len(activations) != len(widths) or any(
        name not in HIDDEN_ACTIVATIONS for name in activations
    ):
        raise ValueError(
            "hidden_activations requires one supported name per hidden layer"
        )
    initializers = _tokens(raw["init_methods"]) if "init_methods" in raw else None
    if initializers is not None and (
        len(initializers) != len(widths) + 1
        or any(name not in {"he", "xavier", "zero"} for name in initializers)
    ):
        raise ValueError(
            "init_methods requires one initializer per hidden and output layer"
        )
    return MLPSettings(training, widths, activations, initializers)


def _ray_settings(raw: Mapping[str, str], base: Path) -> RaySettings:
    _check_keys(raw, {"num_cpus", "cpus_per_task", "timeout_seconds", "temp_directory"})
    cpus = _integer(raw.get("num_cpus", "1"), "num_cpus")
    per_task = _number(raw.get("cpus_per_task", "1"), "cpus_per_task", positive=True)
    if per_task > cpus:
        raise ValueError("cpus_per_task must not exceed num_cpus")
    temporary = raw.get("temp_directory", "").strip()
    directory = (base / temporary).resolve() if temporary else None
    return RaySettings(
        num_cpus=cpus,
        cpus_per_task=per_task,
        timeout_seconds=_number(
            raw.get("timeout_seconds", "300"), "timeout_seconds", positive=True
        ),
        temp_directory=directory,
    )


def load_classifier_settings(
    path: str | Path, *, model_type: str, profile: str | None = None
) -> ClassifierSettings:
    """Validate every declared profile, then return the selected model and resources.

    Section names are case-sensitive. The default profile name is ``model_type``.
    Parsing reads only this INI and never creates directories or starts Ray.
    Invalid syntax or settings raise ValueError; file access errors remain OSError.
    """
    if model_type not in _MODEL_KEYS:
        raise ValueError("Requested model_type must be a supported classifier")
    path = Path(path).resolve()
    # An impossible default-section name makes even empty [DEFAULT] observable.
    parser = _ClassifierParser(
        interpolation=None, strict=True, allow_no_value=False, default_section=""
    )
    try:
        with path.open(encoding="utf-8-sig") as stream:
            parser.read_file(stream)
    except configparser.Error as error:
        raise ValueError("Invalid classifier INI syntax") from error
    if "DEFAULT" in parser:
        raise ValueError("Classifier INI must not declare a DEFAULT section")
    sections: dict[str, Mapping[str, str]] = {
        name: parser[name] for name in parser.sections()
    }
    ray = _ray_settings(sections.get("ray", {}), path.parent)
    models = {
        name: _model_settings(raw) for name, raw in sections.items() if name != "ray"
    }
    selected = model_type if profile is None else profile
    if selected not in models:
        raise ValueError("Requested classifier profile does not exist")
    model = models[selected]
    if model.model_type != model_type:
        raise ValueError(
            "Selected profile model_type does not match requested model_type"
        )
    return ClassifierSettings(ray, model)
