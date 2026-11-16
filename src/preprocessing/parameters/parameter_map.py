"""Read and write the typed XML parameter map used by the CLI."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.preprocessing.parameters.phase import PHASES, PhaseConfig
from src.utils.utils import _boolean, _section


def _decode_value(element: ET.Element):
    if element.tag in {"object", "settings", "parameter"}:
        result = {}
        for entry in element:
            if entry.tag != "entry" or not entry.get("name"):
                raise ValueError("XML objects may contain only named entry elements")
            name = entry.attrib["name"]
            if name in result:
                raise ValueError(f"Duplicate XML entry: {name}")
            if len(entry) != 1:
                raise ValueError(f"XML entry {name} must contain exactly one typed value")
            result[name] = _decode_value(entry[0])
        return result
    if element.tag == "array":
        result = []
        for item in element:
            if item.tag != "item" or len(item) != 1:
                raise ValueError("XML arrays may contain only item elements with one typed value")
            result.append(_decode_value(item[0]))
        return result
    if element.tag == "string":
        return element.text or ""
    if element.tag == "number":
        text = (element.text or "").strip()
        try:
            return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
        except ValueError as error:
            raise ValueError(f"Invalid XML number: {text}") from error
    if element.tag == "boolean":
        text = (element.text or "").strip().casefold()
        if text not in {"true", "false"}:
            raise ValueError(f"Invalid XML boolean: {text}")
        return text == "true"
    if element.tag == "null":
        if element.text and element.text.strip():
            raise ValueError("XML null elements must be empty")
        return None
    raise ValueError(f"Unsupported XML value element: {element.tag}")


def _encode_value(parent: ET.Element, value) -> ET.Element:
    if value is None:
        return ET.SubElement(parent, "null")
    if isinstance(value, bool):
        element = ET.SubElement(parent, "boolean")
        element.text = "true" if value else "false"
        return element
    if isinstance(value, str):
        element = ET.SubElement(parent, "string")
        element.text = value
        return element
    if isinstance(value, int):
        element = ET.SubElement(parent, "number")
        element.text = str(value)
        return element
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("XML parameter maps cannot contain non-finite numbers")
        element = ET.SubElement(parent, "number")
        element.text = repr(value)
        return element
    if isinstance(value, Mapping):
        element = ET.SubElement(parent, "object")
        for name, item in value.items():
            entry = ET.SubElement(element, "entry", {"name": str(name)})
            _encode_value(entry, item)
        return element
    if isinstance(value, (list, tuple)):
        element = ET.SubElement(parent, "array")
        for item in value:
            item_element = ET.SubElement(element, "item")
            _encode_value(item_element, item)
        return element
    raise TypeError(f"Unsupported XML parameter-map value: {type(value).__name__}")


def write_parameter_map(path: str | Path, mapping: Mapping) -> None:
    """Write a mapping as the project's typed XML configuration format."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(to_xml(mapping), encoding="utf-8")


def read_parameter_settings(element: ET.Element) -> tuple[bool, dict]:
    names = {
        "exportTimeSeries": "export_time_series",
        "missingValueTokens": "missing_value_tokens",
        "derivedFeatures": "derived_features",
        "outcomeRules": "patient_outcome_rules",
    }
    if any(
        child.tag == "entry" and child.get("name") in {"parameters", "postoperative_measurements"}
        for child in element
    ):
        raise ValueError("Define parameters and measurements in their phase, not in settings")
    if [child.tag for child in element] != ["selectionEnabled", *names]:
        raise ValueError(
            f"Parameter settings require selectionEnabled, {', '.join(names)} in order"
        )
    values = {}
    for tag, key in names.items():
        child = _section(element, tag)
        if tag == "exportTimeSeries":
            values[key] = _boolean(element, tag)
        elif tag == "missingValueTokens":
            if any(item.tag != "token" for item in child):
                raise ValueError("missingValueTokens accepts only token elements")
            values[key] = [item.text or "" for item in child]
        else:
            if len(child) != 1 or child[0].tag != "object":
                raise ValueError(f"{tag} must contain one typed object")
            values[key] = _decode_value(child[0])
    return _boolean(element, "selectionEnabled"), values


def _decode_phased_map(root: ET.Element) -> dict:
    """Definitions, filters and boundaries live on their owning phase."""
    if [child.tag for child in root] != ["settings", *PHASES]:
        raise ValueError(
            "Embedded parameters requires settings, preOperative, duringOperative, and postOperative in order"
        )
    _, mapping = read_parameter_settings(root[0])
    phase_configs = [PhaseConfig.from_xml(section) for section in root[1:]]
    mapping["included_phases"] = [phase.name for phase in phase_configs]
    mapping["catalog_phases"] = mapping["included_phases"].copy()
    mapping["phase_boundaries"] = {
        phase.boundary: phase.end_event
        for phase in phase_configs
        if phase.boundary != "postoperative_end" or phase.end_event is not None
    }
    mapping["parameters"] = []
    mapping["during_class_names"] = []
    mapping["timeless_class_names"] = []
    measurement_positions = []
    post_class_failures = []
    phases = {tag: values[0] for tag, values in PHASES.items()}
    saw_during_classes = False
    for section in root[1:]:
        phase = phases[section.tag]
        for group in section:
            if group.tag in {"endEvent", "includeIds", "excludeIds"}:
                continue
            if section.tag == "duringOperative" and group.tag == "availableClasses":
                if saw_during_classes:
                    raise ValueError("During-operative availableClasses may appear only once")
                items = list(group)
                if not items or any(
                    item.tag != "className" or not (item.text or "").strip() for item in items
                ):
                    raise ValueError("During-operative availableClasses needs className values")
                class_names = [item.text.strip() for item in items]
                if any(not re.fullmatch(r"epic\.[^.]+", name) for name in class_names):
                    raise ValueError(
                        "During-operative availableClasses must use exact Epic class names"
                    )
                if len({name.casefold() for name in class_names}) != len(class_names):
                    raise ValueError("During-operative availableClasses cannot contain duplicates")
                mapping["during_class_names"] = class_names
                saw_during_classes = True
                continue
            if group.tag == "parameter":
                definitions = [group]
                class_name = None
                timed = True
            elif group.tag == "class":
                class_name = group.get("name", "")
                if not re.fullmatch(r"epic\.[^.]+", class_name):
                    raise ValueError(f"Invalid Epic class: {class_name}")
                flag = group.find("timed")
                if flag is not None and (flag.text or "").strip() not in {"true", "false"}:
                    raise ValueError(f"Class {class_name} timed must be true or false")
                timed = flag is None or flag.text.strip() == "true"
                if not timed:
                    if section.tag != "preOperative":
                        raise ValueError(
                            "Untimed classes are supported only in pre-operative definitions"
                        )
                    mapping["timeless_class_names"].append(class_name)
                definitions = [child for child in group if child.tag == "parameter"]
                if not definitions or len(definitions) + (flag is not None) != len(group):
                    raise ValueError(f"Epic class {class_name} needs parameter definitions")
                failure = group.get("failure")
                if failure is not None:
                    if section.tag != "postOperative" or failure != "nonempty":
                        raise ValueError("Class failure='nonempty' is supported only in post")
                    if class_name in post_class_failures:
                        raise ValueError(f"Duplicate post class failure: {class_name}")
                    post_class_failures.append(class_name)
            else:
                raise ValueError(f"Unsupported definition in parameters/{section.tag}: {group.tag}")
            for element in definitions:
                parameter = _decode_value(element)
                if "id" in parameter or "phase" in parameter:
                    raise ValueError(
                        "Parameter ID and phase belong to the XML structure, not typed entries"
                    )
                parameter_id = element.get("id")
                if not parameter_id:
                    raise ValueError("Parameter IDs must be present and unique")
                if group.tag == "class" and not any(
                    isinstance(field, str)
                    and field.startswith(class_name + ".")
                    and len(field) > len(class_name) + 1
                    for field in parameter.get("source_fields", [])
                ):
                    raise ValueError(
                        f"Parameter {parameter_id} has no source field in {class_name}"
                    )
                has_measurement = "measurement" in parameter
                measurement = parameter.pop("measurement", None)
                position = element.get("measurementOrder")
                if has_measurement and (
                    section.tag != "postOperative"
                    or not isinstance(measurement, dict)
                    or "id" in measurement
                ):
                    raise ValueError(f"Invalid post-operative measurement for {parameter_id}")
                if position is not None and (
                    not has_measurement or not re.fullmatch(r"[1-9]\d*", position)
                ):
                    raise ValueError(f"Invalid measurementOrder for {parameter_id}")
                if has_measurement:
                    measurement = {"id": parameter_id, **measurement}
                    if "source_fields" not in measurement:
                        measurement["source_fields"] = parameter.get("source_fields", [])
                    order = int(position) if position else None
                    if order is not None and any(
                        existing == order for existing, _ in measurement_positions
                    ):
                        raise ValueError(f"Duplicate measurementOrder: {order}")
                    measurement_positions.append((order, measurement))
                if section.tag != "postOperative" and "outcome_criteria" in parameter:
                    raise ValueError(f"Outcome criteria belong in post: {parameter_id}")
                if section.tag == "postOperative":
                    parameter.setdefault("outcome_criteria", {"success": None, "failure": None})
                parameter["id"] = parameter_id
                parameter["phase"] = phase
                parameter["timed"] = timed
                if class_name is not None:
                    parameter["class_name"] = class_name
                mapping["parameters"].append(parameter)
    mapping["postoperative_measurements"] = [
        measurement
        for _, measurement in sorted(
            measurement_positions, key=lambda item: (item[0] is None, item[0] or 0)
        )
    ]
    mapping["post_class_failures"] = post_class_failures
    return mapping


def find_parameters(preprocessing: ET.Element) -> ET.Element:
    if any(child.tag in {"relevantParameters", "parameterSelection"} for child in preprocessing):
        raise ValueError(
            "Use one preprocessing/parameters section containing settings, preOperative, duringOperative, and postOperative"
        )
    return _section(preprocessing, "parameters")


def load_parameter_map_element(root: ET.Element, *, validate_runtime: bool = True) -> dict:
    """Decode the catalog element, whether standalone or embedded in a run XML."""
    if root.tag == "pipelineConfiguration":
        root = find_parameters(_section(root, "preprocessing"))
    if root.tag not in {"parameterMap", "parameters"} or root.get("formatVersion") != "1":
        raise ValueError(
            "Parameter map must have a parameterMap/parameters element with formatVersion='1'"
        )
    if root.tag == "parameters":
        mapping = _decode_phased_map(root)
    else:
        if len(root) != 1 or root[0].tag != "object":
            raise ValueError("Standalone parameter map must contain exactly one object")
        mapping = _decode_value(root[0])
    if validate_runtime:
        _validate_runtime_map(mapping)
    return mapping


def load_parameter_map(path: str | Path, *, validate_runtime: bool = True) -> dict:
    """Load either a standalone parameter map or the catalog embedded in a run XML."""
    source = Path(path)
    try:
        root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"Unable to read XML parameter map {source}: {error}") from error
    return load_parameter_map_element(root, validate_runtime=validate_runtime)


def _validate_runtime_map(mapping: Mapping) -> None:
    if not isinstance(mapping.get("export_time_series", False), bool):
        raise ValueError("export_time_series must be boolean")
    boundaries = mapping.get("phase_boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError(
            "Parameter map needs phase_boundaries for operation_start and operation_end"
        )
    event_names = ["operation_start", "operation_end"]
    if "postoperative_end" in boundaries:
        event_names.append("postoperative_end")
    for event_name in event_names:
        event = boundaries.get(event_name)
        if not isinstance(event, Mapping):
            raise ValueError(f"phase_boundaries.{event_name} needs event_fields and phrases")
        for name in ("event_fields", "phrases"):
            values = event.get(name)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"phase_boundaries.{event_name}.{name} needs nonempty strings")
            if len({" ".join(value.casefold().split()) for value in values}) != len(values):
                raise ValueError(f"phase_boundaries.{event_name}.{name} cannot contain duplicates")
    if {value.casefold().strip() for value in boundaries["operation_start"]["phrases"]} & {
        value.casefold().strip() for value in boundaries["operation_end"]["phrases"]
    }:
        raise ValueError("phase_boundaries operation start and end phrases must not overlap")
    phases = ["Pre-operative", "During-operative", "Post-operative"]
    legacy_phases = ["Pre-operative", "Post-operative"]
    included = mapping.get("included_phases")
    catalog_phases = mapping.get("catalog_phases")
    if included not in (phases, legacy_phases) or catalog_phases not in (phases, legacy_phases):
        raise ValueError(
            "Parameter map phases must be ordered Pre-operative, During-operative, Post-operative"
        )
    if included != catalog_phases or set(included) & set(mapping.get("excluded_phases", [])):
        raise ValueError("Included, catalog, and excluded phases must not conflict")
    parameters = mapping.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError("Parameter map is missing its parameters list")
    required = {
        "clinical_parameter",
        "role",
        "priority",
        "exact_field_names",
        "extraction_definition",
        "parameter_type",
        "unit",
        "value_range",
    }
    seen = set()
    for item in parameters:
        if not isinstance(item, dict) or item.get("phase") not in included:
            raise ValueError("Parameter map contains a parameter outside the supported phases")
        if not required.issubset(item):
            raise ValueError("Parameter map is missing required parameter metadata")
        parameter_id = item.get("id")
        if not parameter_id or parameter_id in seen:
            raise ValueError("Parameter IDs must be present and unique")
        seen.add(parameter_id)
    _validate_duration_specs(mapping, boundaries, parameters)


def _validate_duration_specs(mapping: Mapping, boundaries: Mapping, parameters: list[dict]) -> None:
    by_id = {item["id"]: item for item in parameters}
    for measurement in mapping.get("postoperative_measurements", []):
        duration = measurement.get("duration")
        if duration is None:
            continue
        if not isinstance(duration, Mapping):
            raise ValueError(f"Duration configuration for {measurement['id']} must be an object")
        method = duration.get("method")
        if method not in {"field_interval", "phase_boundary_interval"}:
            raise ValueError(f"Unsupported duration method for {measurement['id']}")
        if not isinstance(duration.get("label"), str) or not duration["label"].strip():
            raise ValueError(f"Duration label is required for {measurement['id']}")
        if duration.get("unit") not in {"seconds", "minutes", "hours", "days"}:
            raise ValueError(f"Unsupported duration unit for {measurement['id']}")
        precision = duration.get("precision")
        if isinstance(precision, bool) or not isinstance(precision, int) or not 0 <= precision <= 6:
            raise ValueError(
                f"Duration precision must be an integer from 0 to 6 for {measurement['id']}"
            )
        if method == "phase_boundary_interval":
            if duration.get("aggregation") != "difference":
                raise ValueError(f"Duration {measurement['id']} must use difference aggregation")
            for name in ("start_boundary", "end_boundary"):
                if duration.get(name) not in boundaries:
                    raise ValueError(
                        f"Duration {measurement['id']}.{name} must name a configured boundary"
                    )
            continue
        if duration.get("aggregation") != "sum_overlaps":
            raise ValueError(f"Duration {measurement['id']} must use sum_overlaps aggregation")
        if duration.get("datetime_format") != "iso8601":
            raise ValueError(f"Duration datetime_format must be iso8601 for {measurement['id']}")
        timezone_name = duration.get("timezone")
        if not isinstance(timezone_name, str) or not timezone_name.strip():
            raise ValueError(f"Duration timezone is required for {measurement['id']}")
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown duration timezone for {measurement['id']}") from error
        parameter = by_id.get(measurement["id"], {})
        sources = parameter.get("source_fields", [])
        for name in ("start_field", "end_field"):
            field = duration.get(name)
            if not isinstance(field, str) or not field.strip() or field not in sources:
                raise ValueError(
                    f"Duration {measurement['id']}.{name} must name a configured source field"
                )
        for name in ("match_start_boundary", "match_end_boundary"):
            if duration.get(name) not in boundaries:
                raise ValueError(
                    f"Duration {measurement['id']}.{name} must name a configured boundary"
                )


def to_xml(mapping: Mapping) -> str:
    """Serialize a mapping to XML text (used by fixtures and tooling)."""
    root = ET.Element(
        "parameterMap",
        {"formatVersion": "1"},
    )
    _encode_value(root, mapping)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"
