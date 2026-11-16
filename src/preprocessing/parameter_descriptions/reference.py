"""Write a definition-only parameter reference."""

from __future__ import annotations

from pathlib import Path


def write_parameter_reference(parameter_map: dict, path: str | Path) -> None:
    """Write configured parameter descriptions for review, without cohort data."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Parameter reference",
        "",
        "Generated from the configured parameter map. It contains definitions, not patient data.",
        "",
        "| ID | Phase | Parameter | Description | Source fields | Type | Unit |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in parameter_map.get("parameters", []):
        source_fields = ", ".join(item.get("source_fields", [])) or "—"
        description = item.get("extraction_definition", "") or item.get("description", "") or "—"
        cells = [
            item.get("id", ""),
            item.get("phase", ""),
            item.get("clinical_parameter", item.get("name", "")),
            description,
            source_fields,
            item.get("parameter_type", ""),
            str(item.get("unit") or "—"),
        ]
        lines.append(
            "| "
            + " | ".join((str(value).replace("|", "\\|").replace("\n", " ") for value in cells))
            + " |"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
