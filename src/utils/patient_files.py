"""Discover explicitly selected CSVs and group their numbered patient parts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


def discover_csv_files(inputs: Iterable[str | Path], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir).resolve()
    excluded = {".git", ".vs", ".venv", "venv", "__pycache__"}
    found: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        candidates = [path] if path.is_file() else path.rglob("*.csv")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.casefold() != ".csv":
                continue
            resolved = candidate.resolve()
            if resolved == output or output in resolved.parents:
                continue
            if any(part.casefold() in excluded for part in resolved.parts):
                continue
            found.add(resolved)
    return sorted(found, key=lambda path: str(path).casefold())


_NUMBERED_PART = re.compile(r"^(?P<family>.+)_(?P<part>[0-9]+)$")


def patient_family(path: Path) -> tuple[str, str]:
    """Split a numbered file part, preserving zero-padded patient identifiers."""
    match = _NUMBERED_PART.fullmatch(path.stem)
    if not match:
        return path.stem, ""
    part = match.group("part")
    if len(part) >= 3 and part.startswith("0"):
        return path.stem, ""
    return match.group("family"), part


def group_patient_files(files: Iterable[Path]) -> list[tuple[Path, ...]]:
    """Group same-directory numbered parts before assigning patient pseudonyms."""
    grouped: dict[tuple[Path, str], list[Path]] = {}
    for path in sorted(files, key=lambda item: (str(item).casefold(), str(item))):
        family, _ = patient_family(path)
        grouped.setdefault((path.parent, family), []).append(path)
    return [tuple(members) for members in grouped.values()]
