"""Parameter definitions with shared settings and phase-owned filters."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.parameters.parameter_map import load_parameter_map_element
from src.preprocessing.parameters.phase import PHASES, PhaseConfig
from src.preprocessing.parameters.settings import (
    ParameterSettingsConfig,
    _filtered_map,
    select_parameter_map,
)
from src.utils.config import XmlConfig
from src.utils.utils import _section


@dataclass(frozen=True)
class ParametersConfig(XmlConfig):
    parameter_map: dict
    settings: ParameterSettingsConfig
    phases: tuple[PhaseConfig, ...]

    @classmethod
    def from_xml(cls, element: ET.Element, project_root: Path) -> ParametersConfig:
        return cls(
            load_parameter_map_element(element),
            ParameterSettingsConfig.from_xml(_section(element, "settings")),
            tuple(PhaseConfig.from_xml(_section(element, tag)) for tag in PHASES),
        )

    def selected_map(self) -> dict:
        if not self.settings.selection_enabled:
            return self.parameter_map
        phases = [phase for phase in self.phases if phase.selected]
        if not phases:
            raise ValueError("Select at least one parameter phase")
        selected = []
        for phase in phases:
            own_ids = {
                item["id"]
                for item in self.parameter_map["parameters"]
                if item["phase"] == phase.name
            }
            if (set(phase.include_ids) | set(phase.exclude_ids)) - own_ids:
                raise ValueError(f"Parameter filters must belong to their {phase.name} phase")
            selected.extend(
                select_parameter_map(
                    self.parameter_map,
                    phases=[phase.name],
                    include_ids=phase.include_ids,
                    exclude_ids=phase.exclude_ids,
                )["parameters"]
            )
        return _filtered_map(self.parameter_map, selected)
