"""Explicit registry of model-specific XML parsers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.models.base import ModelConfig
from src.models.perceptron.config import PerceptronModelConfig


MODEL_CONFIGS: dict[str, type[ModelConfig]] = {"perceptron": PerceptronModelConfig}


def register_model_config(name: str):
    def register(model_type: type[ModelConfig]) -> type[ModelConfig]:
        if name in MODEL_CONFIGS:
            raise ValueError(f"Model configuration already registered: {name}")
        MODEL_CONFIGS[name] = model_type
        model_type.xml_name = name
        return model_type

    return register


def parse_model_config(element: ET.Element, project_root: Path) -> tuple[str, ModelConfig]:
    name = element.get("type", "").strip()
    if not name:
        raise ValueError("Each model requires a type attribute")
    model_type = MODEL_CONFIGS.get(name)
    if model_type is None:
        raise ValueError(f"Unsupported model type: {name}")
    return name, model_type.from_xml(element, project_root)
