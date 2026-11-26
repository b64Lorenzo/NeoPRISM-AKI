"""Typed model option parsers paired with lazily invoked execution handlers."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from src.models.contracts import HIDDEN_ACTIVATIONS, is_forbidden_predictor
from src.pipeline.config import ModelConfig, PreparedModel, parse_integer, parse_number

type OptionValue = int | float | str | tuple[int, ...] | None
type ModelOptions = Mapping[str, OptionValue]


@dataclass(frozen=True)
class ModelStep[OptionsT]:
    config: ModelConfig
    options: OptionsT
    handler: Callable[[ModelConfig, OptionsT], int | None]

    def execute(self) -> int | None:
        return self.handler(self.config, self.options)


class ModelRegistry:
    """Each registration preserves its parser/handler type pairing in a closure."""

    def __init__(self) -> None:
        self._prepare: dict[
            str, Callable[[ModelConfig, Mapping[str, str]], PreparedModel]
        ] = {}

    def register[OptionsT](
        self,
        model_type: str,
        parse_options: Callable[[ModelConfig, Mapping[str, str]], OptionsT],
        handler: Callable[[ModelConfig, OptionsT], int | None],
    ) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", model_type):
            raise ValueError("Model type must match [a-z][a-z0-9_]*")
        if model_type in self._prepare:
            raise ValueError(f"Model type already registered: {model_type}")
        if not callable(parse_options) or not callable(handler):
            raise TypeError(
                "A model registration requires callable option parser and handler"
            )

        def prepare(config: ModelConfig, raw: Mapping[str, str]) -> PreparedModel:
            options = parse_options(config, MappingProxyType(dict(raw)))
            return ModelStep(config, options, handler)

        self._prepare[model_type] = prepare

    def prepare(self, config: ModelConfig, raw: Mapping[str, str]) -> PreparedModel:
        try:
            prepare = self._prepare[config.model_type]
        except KeyError as error:
            raise ValueError(f"Unknown model type: {config.model_type}") from error
        return prepare(config, raw)


def _validate(
    config: ModelConfig,
    raw: Mapping[str, str],
    allowed: set[str],
    targets: set[str],
) -> None:
    if config.target_column not in targets:
        raise ValueError(
            f"Model {config.id} ({config.model_type}) targetColumn must be one of "
            + ", ".join(sorted(targets))
        )
    if any(is_forbidden_predictor(column) for column in config.features):
        raise ValueError(
            "Patient IDs and postoperative/outcome columns cannot be predictor features"
        )
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown option for {config.model_type}: {min(unknown)}")


def _rosenblatt_options(config: ModelConfig, raw: Mapping[str, str]) -> ModelOptions:
    _validate(config, raw, {"epochs", "threshold", "clamped"}, {"aki_label"})
    clamped = raw.get("clamped", "false")
    if clamped not in {"true", "false", "1", "0"}:
        raise ValueError("option clamped must be true, false, 1 or 0")
    return MappingProxyType(
        {
            "epochs": parse_integer(
                raw.get("epochs", "100"), "option epochs", minimum=1
            ),
            "threshold": parse_number(raw.get("threshold", "0"), "option threshold"),
            "clamped": clamped in {"true", "1"},
        }
    )


def _gradient_options(raw: Mapping[str, str]) -> dict[str, OptionValue]:
    return {
        "epochs": parse_integer(raw.get("epochs", "1000"), "option epochs", minimum=1),
        "learning_rate": parse_number(
            raw.get("learning_rate", "0.05"),
            "option learning_rate",
            minimum=0,
            exclusive_minimum=True,
        ),
        "batch_size": parse_integer(
            raw.get("batch_size", "32"), "option batch_size", minimum=1
        ),
        "tolerance": parse_number(
            raw.get("tolerance", "0"), "option tolerance", minimum=0
        ),
    }


def _logistic_options(config: ModelConfig, raw: Mapping[str, str]) -> ModelOptions:
    _validate(
        config,
        raw,
        {"epochs", "learning_rate", "batch_size", "tolerance", "l2"},
        {"aki_label", "aki_stage"},
    )
    options = _gradient_options(raw)
    options["l2"] = parse_number(raw.get("l2", "0"), "option l2", minimum=0)
    return MappingProxyType(options)


def _multilayer_options(config: ModelConfig, raw: Mapping[str, str]) -> ModelOptions:
    _validate(
        config,
        raw,
        {
            "epochs",
            "learning_rate",
            "batch_size",
            "tolerance",
            "hidden_layers",
            "activation",
        },
        {"aki_label", "aki_stage"},
    )
    options = _gradient_options(raw)
    options["hidden_layers"] = tuple(
        parse_integer(value.strip(), "option hidden_layers", minimum=1)
        for value in raw.get("hidden_layers", "16").split(",")
    )
    activation = raw.get("activation", "relu")
    if activation not in HIDDEN_ACTIVATIONS:
        raise ValueError(
            "option activation must be one of: " + ", ".join(HIDDEN_ACTIVATIONS)
        )
    options["activation"] = activation
    return MappingProxyType(options)


def _fit_model(config: ModelConfig, options: ModelOptions) -> int:
    # This import must not make disabled templates depend on a training backend.
    from src.models.classification import fit

    fit(
        config.input_matrix,
        config.output_dir,
        model_type=config.model_type,
        target_column=config.target_column,
        features=config.features,
        train_fraction=config.train_fraction,
        seed=config.seed,
        **options,
    )
    return 0


def create_default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("rosenblatt_perceptron", _rosenblatt_options, _fit_model)
    registry.register("logistic_regression", _logistic_options, _fit_model)
    registry.register("multilayer_perceptron", _multilayer_options, _fit_model)
    return registry
