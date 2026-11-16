"""Root logging configuration."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import XmlConfig
from src.utils.utils import _path, _required


@dataclass(frozen=True)
class LoggingConfig(XmlConfig):
    directory: Path
    level: str

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> LoggingConfig:
        level = _required(element, "level").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("Logging level must be DEBUG, INFO, WARNING, or ERROR")
        return cls(_path(project_root, _required(element, "directory")), level)
