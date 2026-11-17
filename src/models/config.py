"""Parse an optional collection of model instances."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.models.base import ModelConfig
from src.models.registry import parse_model_config
from src.utils.config import XmlConfig
from src.utils.utils import _boolean


@dataclass(frozen=True)
class ModelInstanceConfig(XmlConfig):
    id: str
    type: str
    run: bool
    config: ModelConfig

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> ModelInstanceConfig:
        instance_id = element.get("id", "").strip()
        if not instance_id:
            raise ValueError("Each model requires a nonempty id attribute")
        model_type, model_config = parse_model_config(element, project_root)
        return cls(instance_id, model_type, _boolean(element, "run"), model_config)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "run": self.run,
            "config": self.config.as_dict(),
        }


@dataclass(frozen=True)
class ModelsConfig(XmlConfig):
    models: tuple[ModelInstanceConfig, ...]

    @classmethod
    def from_xml(cls, element: ET.Element | None, project_root: Path) -> ModelsConfig:
        if element is None:
            return cls(())
        models = tuple(ModelInstanceConfig.from_xml(item, project_root) for item in element.findall("model"))
        ids = [model.id for model in models]
        if len(ids) != len(set(ids)):
            raise ValueError("Model instance IDs must be unique")
        return cls(models)

    def as_dict(self) -> dict:
        return {"models": [model.as_dict() for model in self.models]}
