"""Typed dimensions for the patient timeline dashboard."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardLayout:
    chart_height: int = 740
    panel_max_height: int = 320

    def __post_init__(self) -> None:
        if not 320 <= self.chart_height <= 900:
            raise ValueError("chartHeight must be between 320 and 900 pixels")
        if not 120 <= self.panel_max_height <= 900:
            raise ValueError("panelMaxHeight must be between 120 and 900 pixels")

    @classmethod
    def from_xml(cls, element: ET.Element | None) -> DashboardLayout:
        if element is None:
            raise ValueError("Missing patientTimeline/layout section")
        values = []
        for name in ("chartHeight", "panelMaxHeight"):
            node = element.find(name)
            if node is None or node.text is None:
                raise ValueError(f"Missing layout/{name}")
            try:
                values.append(int(node.text.strip()))
            except ValueError as error:
                raise ValueError(f"layout/{name} must be an integer") from error
        return cls(*values)
