"""Small shared building blocks for typed XML component settings."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path

from src.utils.utils import _boolean, _path, _required


@dataclass(frozen=True)
class XmlConfig:
    def as_dict(self) -> dict:
        def project(value):
            if isinstance(value, XmlConfig):
                return value.as_dict()
            return list(value) if isinstance(value, list) else value

        return {field.name: project(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True)
class OutputConfig(XmlConfig):
    run: bool
    output: Path

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path):
        return cls(_boolean(element, "run"), _path(project_root, _required(element, "output")))
