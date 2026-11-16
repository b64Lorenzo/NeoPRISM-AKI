"""Explicit execution handlers, separate from model configuration parsing."""

from collections.abc import Callable

from src.models.base import ModelConfig
from src.models.perceptron.config import PerceptronModelConfig
from src.models.perceptron.runner import run_perceptron

MODEL_RUNNERS: dict[type[ModelConfig], Callable[..., list[str]]] = {
    PerceptronModelConfig: run_perceptron,
}


def get_model_runner(config: ModelConfig) -> Callable[..., list[str]]:
    runner = MODEL_RUNNERS.get(type(config))
    if runner is None:
        raise ValueError(f"No runner is registered for model: {config.xml_name}")
    return runner
