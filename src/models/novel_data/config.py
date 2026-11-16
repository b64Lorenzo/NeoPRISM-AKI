"""Saved-model prediction inputs and outputs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import XmlConfig
from src.utils.utils import _path, _required


@dataclass(frozen=True)
class NovelDataConfig(XmlConfig):
    """Saved-model prediction inputs and outputs."""

    vectors: Path
    model_input: Path
    predictions: Path

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> NovelDataConfig:
        return cls(
            vectors=_path(project_root, _required(element, "vectors")),
            model_input=_path(project_root, _required(element, "modelInput")),
            predictions=_path(project_root, _required(element, "predictions")),
        )
