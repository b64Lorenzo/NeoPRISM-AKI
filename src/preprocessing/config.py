"""Preprocessing configuration delegates each structured child to its owner."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.csv_header_descriptions.config import CsvHeaderDescriptionsConfig
from src.preprocessing.parameter_descriptions.config import ParameterDescriptionsConfig
from src.preprocessing.parameters.config import ParametersConfig
from src.preprocessing.parameters.parameter_map import find_parameters
from src.preprocessing.plotting.config import PreprocessingPlotConfig
from src.utils.config import XmlConfig
from src.utils.utils import _boolean, _path, _required, _section


@dataclass(frozen=True)
class PreprocessingConfig(XmlConfig):
    run: bool
    input_directory: Path
    output_directory: Path
    parameters: ParametersConfig
    parameter_descriptions: ParameterDescriptionsConfig
    csv_header_descriptions: CsvHeaderDescriptionsConfig
    plotting: PreprocessingPlotConfig

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> PreprocessingConfig:
        result = cls(
            _boolean(element, "run"),
            _path(project_root, _required(element, "inputDirectory")),
            _path(project_root, _required(element, "outputDirectory")),
            ParametersConfig.from_xml(find_parameters(element), project_root),
            ParameterDescriptionsConfig.from_xml(
                _section(element, "parameterDescriptions"), project_root
            ),
            CsvHeaderDescriptionsConfig.from_xml(
                _section(element, "csvHeaderDescriptions"), project_root
            ),
            PreprocessingPlotConfig.from_xml(_section(element, "plotting"), project_root),
        )
        if (
            result.plotting.time_series.run
            and result.run
            and not result.parameters.parameter_map.get("export_time_series", False)
        ):
            raise ValueError(
                "Time-series plotting with preprocessing requires export_time_series=true"
            )
        return result

    def as_dict(self) -> dict:
        result = super().as_dict()
        result.pop("parameters")
        result["parameter_map"] = self.parameters.parameter_map
        return result
