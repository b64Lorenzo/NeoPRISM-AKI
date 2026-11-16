"""Select a registered model without parsing algorithm-specific children."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.models.base import ModelConfig
from src.models.registry import parse_active_model
from src.utils.config import XmlConfig


@dataclass(frozen=True)
class ModelsConfig(XmlConfig):
    active_model: str
    active_config: ModelConfig

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> ModelsConfig:
        return cls(*parse_active_model(element, project_root))

    def as_dict(self) -> dict:
        return {
            "active_model": self.active_model,
            "models": {self.active_model: self.active_config.as_dict()},
            "active_config": self.active_config,
        }
