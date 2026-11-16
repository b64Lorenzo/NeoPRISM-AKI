"""Model configuration interface, independent of concrete algorithms."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

from src.utils.config import XmlConfig


class ModelConfig(XmlConfig, ABC):
    xml_name: str

    @classmethod
    @abstractmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> ModelConfig:
        raise NotImplementedError
