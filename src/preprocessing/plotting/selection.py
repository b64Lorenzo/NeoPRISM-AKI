"""Shared phase, parameter and class selection for patient-level plots."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import XmlConfig
from src.utils.utils import _ids, _list, _required, _section

PHASES = {
    "all": None,
    "preoperative": "Pre-operative",
    "duringoperative": "During-operative",
    "postoperative": "Post-operative",
}


@dataclass(frozen=True)
class PlotSelection(XmlConfig):
    phase: str
    parameter_ids: list[str]
    class_names: list[str]

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(
                "Plot phase must be all, preoperative, duringoperative, or postoperative"
            )

    def select(self, parameter_map: Mapping) -> list[dict]:
        parameters = parameter_map.get("parameters", [])
        ids, classes = set(self.parameter_ids), set(self.class_names)
        unknown_ids = ids - {item["id"] for item in parameters}
        unknown_classes = classes - {item.get("class_name") for item in parameters}
        if unknown_ids or unknown_classes:
            raise ValueError(
                f"Unknown plot parameters/classes: {sorted(unknown_ids | unknown_classes)}"
            )
        phase = PHASES[self.phase]
        phases = set(PHASES.values()) - {None}
        selected = [
            item
            for item in parameters
            if item.get("phase") in phases
            and (phase is None or item["phase"] == phase)
            and (not ids or item["id"] in ids)
            and (not classes or item.get("class_name") in classes)
        ]
        if (ids or classes) and not selected:
            raise ValueError("No parameters match plot selection")
        return selected

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> PlotSelection:
        for container, tag in (("parameterIds", "id"), ("classNames", "name")):
            if any(
                item.tag != tag or len(item) or not (item.text or "").strip()
                for item in _section(element, container)
            ):
                raise ValueError(f"{container} requires nonempty {tag} values")
        return cls(
            _required(element, "phase"),
            _ids(element, "parameterIds"),
            _list(element, "classNames", "name"),
        )
