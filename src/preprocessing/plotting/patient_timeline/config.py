"""Patient timeline selection and dashboard output settings."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.plotting.patient_timeline.layout.config import DashboardLayout
from src.utils.config import XmlConfig
from src.utils.utils import _boolean, _ids, _path, _required, _section


@dataclass(frozen=True)
class PatientTimelineConfig(XmlConfig):
    run: bool
    patient_ids: list[str]
    compare_all_patients: bool
    include_study_plots: bool
    layout: DashboardLayout
    output: Path

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> PatientTimelineConfig:
        return cls(
            _boolean(element, "run"),
            _ids(element, "patientIds"),
            _boolean(element, "compareAllPatients"),
            _boolean(element, "includeStudyPlots"),
            DashboardLayout.from_xml(_section(element, "layout")),
            _path(project_root, _required(element, "output")),
        )
