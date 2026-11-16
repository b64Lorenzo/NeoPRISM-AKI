"""Compact shared settings and reusable catalog filtering."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from src.preprocessing.parameters.parameter_map import read_parameter_settings
from src.utils.config import XmlConfig


@dataclass(frozen=True)
class ParameterSettingsConfig(XmlConfig):
    selection_enabled: bool
    values: dict

    @classmethod
    def from_xml(cls, element: ET.Element) -> ParameterSettingsConfig:
        return cls(*read_parameter_settings(element))


def select_parameter_map(
    parameter_map: dict, *, phases: list[str], include_ids: list[str], exclude_ids: list[str]
) -> dict:
    """Return a copy containing only the configured phases and parameter IDs."""
    all_parameters = parameter_map.get("parameters", [])
    known_ids = {item["id"] for item in all_parameters}
    requested = set(include_ids) | set(exclude_ids)
    unknown = sorted(requested - known_ids)
    if unknown:
        raise ValueError(f"Unknown parameter ID: {', '.join(unknown)}")
    overlap = set(include_ids) & set(exclude_ids)
    if overlap:
        raise ValueError(
            f"Parameter IDs cannot be both included and excluded: {', '.join(sorted(overlap))}"
        )
    selected = [
        item
        for item in all_parameters
        if item.get("phase") in phases
        and (not include_ids or item["id"] in include_ids)
        and (item["id"] not in exclude_ids)
    ]
    if include_ids and {item["id"] for item in selected} != set(include_ids):
        raise ValueError("Every included parameter ID must belong to a selected phase")
    return _filtered_map(parameter_map, selected)


def _filtered_map(parameter_map: dict, selected: list[dict]) -> dict:
    result = dict(parameter_map)
    result["parameters"] = selected
    if "postoperative_measurements" in result:
        postoperative_ids = {
            item["id"] for item in selected if item.get("phase") == "Post-operative"
        }
        result["postoperative_measurements"] = [
            item
            for item in result["postoperative_measurements"]
            if item.get("id") in postoperative_ids
        ]
    return result
