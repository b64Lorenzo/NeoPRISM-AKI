"""Patient missingness plot selection and destination."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.plotting.selection import PlotSelection
from src.utils.utils import _boolean, _path, _required


@dataclass(frozen=True)
class MissingnessByPatientConfig(PlotSelection):
    run: bool
    output: Path

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path):
        return cls(
            **PlotSelection.from_xml(element, project_root).as_dict(),
            run=_boolean(element, "run"),
            output=_path(project_root, _required(element, "output")),
        )
