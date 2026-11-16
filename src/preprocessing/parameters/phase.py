"""Phase-owned boundaries and optional parameter filters."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from src.utils.utils import _ids, _list, _section

PHASES = {
    "preOperative": ("Pre-operative", "operation_start"),
    "duringOperative": ("During-operative", "operation_end"),
    "postOperative": ("Post-operative", "postoperative_end"),
}


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    boundary: str
    selected: bool
    include_ids: list[str]
    exclude_ids: list[str]
    end_event: dict | None

    @classmethod
    def from_xml(cls, element: ET.Element) -> PhaseConfig:
        name, boundary = PHASES[element.tag]
        flag = element.get("selected", "true")
        if flag not in {"true", "false", "1", "0"}:
            raise ValueError(f"{element.tag} selected must be true or false")
        event = _section(element, "endEvent")
        if len(event) and [child.tag for child in event] != ["fields", "phrases"]:
            raise ValueError(f"phase_boundaries.{boundary} needs event_fields and phrases")
        end_event = (
            {
                "event_fields": _list(event, "fields", "field"),
                "phrases": _list(event, "phrases", "phrase"),
            }
            if len(event)
            else None
        )
        return cls(
            name,
            boundary,
            flag in {"true", "1"},
            _ids(element, "includeIds") if element.find("includeIds") is not None else [],
            _ids(element, "excludeIds") if element.find("excludeIds") is not None else [],
            end_event,
        )
