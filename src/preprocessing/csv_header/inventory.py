
"""Inventory CSV headers without reading patient data rows.

Generates:
    - CSV inventory of headers and their positions
    - Markdown summary of files, schemas, patient families,
      header frequencies and Epic classes

No XML configuration or clinical phase inference is used.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from html import escape
from io import StringIO
from pathlib import Path
from typing import Iterable

from src.utils.patient_files import (
    discover_csv_files,
    patient_family,
)


_MAX_HEADER_BYTES = 1_000_000


# ---------------------------------------------------------------------
# CSV header parsing
# ---------------------------------------------------------------------

def _header_separators(sample: str) -> tuple[Counter, bool]:
    """Count delimiters outside quotes and detect open quotes."""
    counts = Counter()
    in_quotes = False
    index = 0

    while index < len(sample):
        char = sample[index]

        if char == '"':
            if (
                in_quotes
                and index + 1 < len(sample)
                and sample[index + 1] == '"'
            ):
                index += 1
            else:
                in_quotes = not in_quotes

        elif not in_quotes and char in ",;\t|":
            counts[char] += 1

        index += 1

    return counts, in_quotes


def _delimiter_from_header(sample: str) -> str:
    """Determine the CSV delimiter outside quoted fields."""
    counts, _ = _header_separators(sample)

    for delimiter in ("\t", ";", ",", "|"):
        if counts[delimiter]:
            return delimiter

    return ","


def _read_header(path: Path) -> list[str]:
    """Read only the first CSV record, including quoted newlines."""
    with path.open("rb") as stream:
        first_line = stream.readline(_MAX_HEADER_BYTES + 1)

        if len(first_line) > _MAX_HEADER_BYTES:
            raise ValueError(
                f"CSV header exceeds supported size: {path}"
            )

        if not first_line:
            raise ValueError(f"Empty CSV header: {path}")

        try:
            sample = first_line.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            sample = first_line.decode("cp1252")
            encoding = "cp1252"

        sample_bytes = first_line

        for _ in range(20):
            if not _header_separators(sample)[1]:
                break

            continuation = stream.readline(
                _MAX_HEADER_BYTES - len(sample_bytes) + 1
            )

            if len(sample_bytes) + len(continuation) > _MAX_HEADER_BYTES:
                raise ValueError(
                    f"CSV header exceeds supported size: {path}"
                )

            if not continuation:
                raise ValueError(
                    f"Unterminated quoted CSV header: {path}"
                )

            sample_bytes += continuation
            sample = sample_bytes.decode(encoding)

        else:
            raise ValueError(
                f"CSV header has too many physical lines: {path}"
            )

        delimiter = _delimiter_from_header(sample)

        header = next(
            csv.reader(
                StringIO(sample, newline=""),
                delimiter=delimiter,
                strict=True,
            ),
            [],
        )

        if not header or all(not field for field in header):
            raise ValueError(f"Empty CSV header: {path}")

        return header


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def _markdown(value: object) -> str:
    """Escape text for a Markdown table cell."""
    text = (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
    )

    return escape(
        "".join(
            char if char.isprintable() else " "
            for char in text
        )
    )


def _epic_class(field: str) -> str | None:
    """Extract the epic.CLASS prefix of a field."""
    parts = field.split(".", 2)

    if (
        len(parts) == 3
        and parts[0] == "epic"
        and parts[1]
        and parts[2]
    ):
        return ".".join(parts[:2])

    return None


# ---------------------------------------------------------------------
# Main inventory operation
# ---------------------------------------------------------------------

def write_csv_header_descriptions(
    inputs: Iterable[str | Path],
    output_directory: str | Path,
    inventory_output: str | Path,
    summary_output: str | Path,
) -> dict[str, int]:
    """Generate header inventory and Markdown summary.

    Patient families are identified using same-directory
    filename siblings, rather than suffix matching alone.

    Patient data rows are never loaded.
    """
    sources = [Path(item).resolve() for item in inputs]

    if not sources:
        raise ValueError("At least one CSV input is required")

    output_dir = Path(output_directory).resolve()
    inventory = Path(inventory_output).resolve()
    summary = Path(summary_output).resolve()

    if inventory == summary:
        raise ValueError(
            "Inventory and summary output paths must differ"
        )

    paths = discover_csv_files(sources, output_dir)

    if inventory in paths or summary in paths:
        raise ValueError(
            "Report output overlaps a CSV input"
        )

    if not paths:
        raise ValueError("No CSV files found")

    # -------------------------------------------------------------
    # Build sibling filename index
    # -------------------------------------------------------------

    stems_by_parent: dict[Path, set[str]] = defaultdict(set)

    for path in paths:
        stems_by_parent[path.parent].add(path.stem)

    # -------------------------------------------------------------
    # Read headers and identify patient families
    # -------------------------------------------------------------

    files: list[dict] = []

    for path in paths:
        source_index = None
        relative = None

        for index, source in enumerate(sources, start=1):
            if path == source:
                source_index = index
                relative = path.name
                break

            if source.is_dir() and path.is_relative_to(source):
                source_index = index
                relative = path.relative_to(source).as_posix()
                break

        if source_index is None or relative is None:
            raise ValueError(
                f"CSV outside configured inputs: {path}"
            )

        base, part = patient_family(
            path,
            stems_by_parent[path.parent],
        )

        family_key = (path.parent, base)

        relative_parent = Path(relative).parent
        family_name = (
            (relative_parent / base).as_posix()
            if relative_parent != Path(".")
            else base
        )

        files.append({
            "source": source_index,
            "file": relative,
            "family": family_name,
            "family_key": family_key,
            "part": part,
            "path": path,
            "headers": _read_header(path),
        })

    files.sort(
        key=lambda item: (
            item["source"],
            item["file"].casefold(),
            item["file"],
        )
    )

    # -------------------------------------------------------------
    # Identify ordered schemas and header frequencies
    # -------------------------------------------------------------

    schemas: dict[tuple[str, ...], str] = {}
    schema_files = defaultdict(list)
    families = defaultdict(list)
    frequency = Counter()

    for item in files:
        headers = item["headers"]
        schema_key = tuple(headers)

        if schema_key not in schemas:
            schemas[schema_key] = f"S{len(schemas) + 1:03d}"

        schema = schemas[schema_key]
        item["schema"] = schema

        schema_files[schema].append(item)
        families[item["family_key"]].append(item)

        # Each header is counted at most once per file.
        frequency.update(set(headers))

    shared_headers = sum(
        count == len(files)
        for count in frequency.values()
    )

    # -------------------------------------------------------------
    # Build Epic class statistics
    # -------------------------------------------------------------

    class_headers: dict[str, set[str]] = defaultdict(set)
    class_files: dict[str, set[Path]] = defaultdict(set)
    class_families: dict[str, set[tuple[Path, str]]] = (
        defaultdict(set)
    )

    for item in files:
        for header in set(item["headers"]):
            epic_class = _epic_class(header)

            if epic_class is None:
                continue

            class_headers[epic_class].add(header)
            class_files[epic_class].add(item["path"])
            class_families[epic_class].add(item["family_key"])

    # -------------------------------------------------------------
    # Markdown overview
    # -------------------------------------------------------------

    lines = [
        "# CSV header summary",
        "",
        "Header-only inspection. No patient data rows "
        "or outcomes are read.",
        "",
        "File names may identify patients. "
        "Keep this report private.",
        "",
        f"- CSV files: {len(files)}",
        f"- Patient families: {len(families)}",
        f"- Distinct headers: {len(frequency)}",
        f"- Headers in every file: {shared_headers}",
        f"- Distinct ordered schemas: {len(schemas)}",
        f"- Distinct Epic classes: {len(class_headers)}",
        "",
        "## Files",
        "",
        "| Input | CSV file | Patient family | Part | Columns | Schema |",
        "|---|---|---|---|---:|---|",
    ]

    for item in files:
        values = (
            item["source"],
            item["file"],
            item["family"],
            item["part"] or "—",
            len(item["headers"]),
            item["schema"],
        )

        lines.append(
            "| "
            + " | ".join(_markdown(value) for value in values)
            + " |"
        )

    # -------------------------------------------------------------
    # Ordered header schemas
    # -------------------------------------------------------------

    lines.extend([
        "",
        "## Ordered header schemas",
        "",
        "Two files share a schema only if all column names "
        "and their positions are identical.",
        "",
        "| Schema | Files | CSV file names |",
        "|---|---:|---|",
    ])

    for schema, members in schema_files.items():
        names = ", ".join(
            f"input {item['source']}: {item['file']}"
            for item in members
        )

        values = (
            schema,
            len(members),
            names,
        )

        lines.append(
            "| "
            + " | ".join(_markdown(value) for value in values)
            + " |"
        )

    # -------------------------------------------------------------
    # Patient family summary
    # -------------------------------------------------------------

    lines.extend([
        "",
        "## Patient filename families",
        "",
        "Numbered files are grouped only when the base "
        "file exists or multiple numbered sibling files "
        "share the same base in the same directory.",
        "",
        "Zero-padded suffixes such as `_01` are not "
        "interpreted as file parts.",
        "",
        "| Input(s) | Patient family | Files | CSV file names |",
        "|---|---|---:|---|",
    ])

    sorted_families = sorted(
        families.values(),
        key=lambda members: (
            members[0]["source"],
            members[0]["family"].casefold(),
        ),
    )

    for members in sorted_families:
        source_ids = ", ".join(
            str(value)
            for value in sorted({
                item["source"]
                for item in members
            })
        )

        values = (
            source_ids,
            members[0]["family"],
            len(members),
            ", ".join(
                item["file"] for item in members
            ),
        )

        lines.append(
            "| "
            + " | ".join(_markdown(value) for value in values)
            + " |"
        )

    # -------------------------------------------------------------
    # Header presence summary
    # -------------------------------------------------------------

    lines.extend([
        "",
        "## Header presence",
        "",
        "Each header is counted once per CSV file, "
        "even if it appears more than once in that file.",
        "",
        "| Header | CSV files | Epic class |",
        "|---|---:|---|",
    ])

    for header, count in sorted(
        frequency.items(),
        key=lambda entry: entry[0].casefold(),
    ):
        values = (
            header,
            count,
            _epic_class(header) or "—",
        )

        lines.append(
            "| "
            + " | ".join(_markdown(value) for value in values)
            + " |"
        )

    # -------------------------------------------------------------
    # Epic class index
    # -------------------------------------------------------------

    lines.extend([
        "",
        "## Epic class index",
        "",
        "Epic classes are extracted from headers matching "
        "`epic.CLASS.item`.",
        "",
        "Distinct headers, CSV files and patient families "
        "are counted independently.",
        "",
        "| Epic class | Distinct headers | CSV files | Patient families |",
        "|---|---:|---:|---:|",
    ])

    for epic_class in sorted(
        class_headers,
        key=str.casefold,
    ):
        values = (
            epic_class,
            len(class_headers[epic_class]),
            len(class_files[epic_class]),
            len(class_families[epic_class]),
        )

        lines.append(
            "| "
            + " | ".join(_markdown(value) for value in values)
            + " |"
        )

    if not class_headers:
        lines.append("| — | 0 | 0 | 0 |")

    # -------------------------------------------------------------
    # Write CSV inventory
    # -------------------------------------------------------------

    inventory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with inventory.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)

        writer.writerow([
            "source",
            "file",
            "family",
            "part",
            "position",
            "header",
        ])

        for item in files:
            for position, header in enumerate(
                item["headers"],
                start=1,
            ):
                writer.writerow([
                    item["source"],
                    item["file"],
                    item["family"],
                    item["part"],
                    position,
                    header,
                ])

    # -------------------------------------------------------------
    # Write Markdown summary
    # -------------------------------------------------------------

    summary.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Return statistics
    # -------------------------------------------------------------

    return {
        "files": len(files),
        "families": len(families),
        "unique_headers": len(frequency),
        "shared_headers": shared_headers,
        "schemas": len(schemas),
        "epic_classes": len(class_headers),
    }
