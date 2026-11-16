"""Distribution plot execution, histogram and selection settings."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.plotting.selection import PlotSelection
from src.utils.utils import _boolean, _required


@dataclass(frozen=True)
class DistributionPlotConfig(PlotSelection):
    run: bool
    bins: int

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path):
        bins = int(_required(element, "bins"))
        if bins < 1:
            raise ValueError("Distribution bins must be positive")
        return cls(
            **PlotSelection.from_xml(element, project_root).as_dict(),
            run=_boolean(element, "run"),
            bins=bins,
        )
