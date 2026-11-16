"""Shared XML value readers used by component configurations."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _required(parent: ET.Element, tag: str) -> str:
    elements = parent.findall(tag)
    if len(elements) > 1:
        raise ValueError(f"Duplicate XML value: {tag}")
    element = elements[0] if elements else None
    if element is None or element.text is None or not element.text.strip():
        raise ValueError(f"Missing required XML value: {tag}")
    return element.text.strip()


def _boolean(parent: ET.Element, tag: str) -> bool:
    value = _required(parent, tag).casefold()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{tag} must be true or false")
    return value in {"true", "1"}


def _path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _section(parent: ET.Element, tag: str) -> ET.Element:
    sections = parent.findall(tag)
    if not sections:
        raise ValueError(f"Missing required XML section: {tag}")
    if len(sections) != 1:
        raise ValueError(f"Duplicate XML section: {tag}")
    return sections[0]


def _list(parent: ET.Element, container: str, item_tag: str = "id") -> list[str]:
    element = _section(parent, container)
    values = [
        item.text.strip() for item in element.findall(item_tag) if item.text and item.text.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate value in {container}")
    return values


def _ids(parent: ET.Element, container: str) -> list[str]:
    return _list(parent, container)
