"""Header-only CSV inventory settings."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import XmlConfig
from src.utils.utils import _boolean, _path, _required


@dataclass(frozen=True)
class CsvHeaderDescriptionsConfig(XmlConfig):
    run: bool
    inventory_output: Path
    summary_output: Path

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> CsvHeaderDescriptionsConfig:
        return cls(
            _boolean(element, "run"),
            _path(project_root, _required(element, "inventoryOutput")),
            _path(project_root, _required(element, "summaryOutput")),
        )
