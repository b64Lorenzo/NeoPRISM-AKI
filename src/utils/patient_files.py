
"""Discover CSV files and identify patient file families."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_EXCLUDED = {
    ".git", ".vs", ".venv", "venv", "__pycache__",
}

_NUMBERED_PART = re.compile(r"^(.+)_([0-9]{1,2})$")


def discover_csv_files(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
) -> list[Path]:
    """Discover CSV files, excluding the report directory."""
    output = Path(output_dir).resolve()
    found: set[Path] = set()

    for item in inputs:
        path = Path(item).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Input path does not exist: {path}"
            )

        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("*")
        else:
            raise ValueError(f"Invalid input path: {path}")

        for candidate in candidates:
            if not candidate.is_file():
                continue

            if candidate.suffix.casefold() != ".csv":
                continue

            resolved = candidate.resolve()

            if resolved == output or output in resolved.parents:
                continue

            if any(
                part.casefold() in _EXCLUDED
                for part in resolved.parts
            ):
                continue

            found.add(resolved)

    return sorted(
        found,
        key=lambda path: (str(path).casefold(), str(path)),
    )


def patient_family(
    path: Path,
    sibling_stems: set[str] | None = None,
) -> tuple[str, str]:
    """Identify a patient family using same-directory siblings.

    A suffix _1 through _99 is treated as a numbered part
    only when the base file exists or at least two numbered
    sibling parts share the same base.

    Zero-padded suffixes such as _01 are not treated as parts.
    """
    stem = path.stem
    match = _NUMBERED_PART.fullmatch(stem)

    if match is None:
        return stem, ""

    base, part = match.groups()

    if len(part) > 1 and part.startswith("0"):
        return stem, ""

    if sibling_stems is None:
        return stem, ""

    numbered_parts = {
        candidate.group(2)
        for sibling in sibling_stems
        if (
            candidate := _NUMBERED_PART.fullmatch(sibling)
        ) is not None
        and candidate.group(1) == base
        and not (
            len(candidate.group(2)) > 1
            and candidate.group(2).startswith("0")
        )
    }

    if base not in sibling_stems and len(numbered_parts) < 2:
        return stem, ""

    return base, part


def group_patient_files(
    files: Iterable[Path],
) -> list[tuple[Path, ...]]:
    """Group files into patient families within each directory."""
    paths = sorted(
        (Path(path).resolve() for path in files),
        key=lambda path: (str(path).casefold(), str(path)),
    )

    stems_by_parent: dict[Path, set[str]] = {}

    for path in paths:
        stems_by_parent.setdefault(
            path.parent, set()
        ).add(path.stem)

    grouped: dict[tuple[Path, str], list[Path]] = {}

    for path in paths:
        family, _ = patient_family(
            path,
            stems_by_parent[path.parent],
        )

        grouped.setdefault(
            (path.parent, family), []
        ).append(path)

    return [
        tuple(members)
        for members in grouped.values()
    ]
