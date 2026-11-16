"""Time-series plot grouping, filters and output settings."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import XmlConfig
from src.utils.utils import _boolean, _ids, _list, _path, _required


@dataclass(frozen=True)
class TimeSeriesPlotConfig(XmlConfig):
    run: bool
    group_by: str
    parameter_ids: list[str]
    class_names: list[str]
    patient_ids: list[str]
    output_directory: Path

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> TimeSeriesPlotConfig:
        group_by = _required(element, "groupBy")
        if group_by not in {"parameter", "class"}:
            raise ValueError("Time-series groupBy must be parameter or class")
        return cls(
            _boolean(element, "run"),
            group_by,
            _ids(element, "parameterIds"),
            _list(element, "classNames", "name"),
            _ids(element, "patientIds"),
            _path(project_root, _required(element, "outputDirectory")),
        )
