"""Preprocessing plot settings composed from independently controlled children."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.plotting.distributions.config import DistributionPlotConfig
from src.preprocessing.plotting.missingness_by_patient.config import MissingnessByPatientConfig
from src.preprocessing.plotting.patient_timeline.config import PatientTimelineConfig
from src.preprocessing.plotting.time_series.config import TimeSeriesPlotConfig
from src.utils.config import XmlConfig
from src.utils.utils import _boolean, _required, _section


@dataclass(frozen=True)
class PreprocessingPlotConfig(XmlConfig):
    run: bool
    distributions: DistributionPlotConfig
    theme: str
    missingness_by_patient: MissingnessByPatientConfig
    time_series: TimeSeriesPlotConfig
    patient_timeline: PatientTimelineConfig

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> PreprocessingPlotConfig:
        if any(child.tag in {"phase", "bins"} for child in element):
            raise ValueError("Put distribution phase and bins inside plotting/distributions")
        theme = _required(element, "theme")
        if theme not in {"light", "dark"}:
            raise ValueError("Preprocessing plotting theme must be light or dark")
        return cls(
            _boolean(element, "run"),
            DistributionPlotConfig.from_xml(_section(element, "distributions"), project_root),
            theme,
            MissingnessByPatientConfig.from_xml(
                _section(element, "missingnessByPatient"), project_root
            ),
            TimeSeriesPlotConfig.from_xml(_section(element, "timeSeries"), project_root),
            PatientTimelineConfig.from_xml(_section(element, "patientTimeline"), project_root),
        )
