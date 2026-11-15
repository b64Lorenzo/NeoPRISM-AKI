"""Load the run-control XML and prepare configured parameter selections."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from src.parameter_map import load_parameter_map_element
from src.model_config import parse_active_model
from src.timeline_dashboard_view import DashboardLayout

PHASES = ("Pre-operative", "During-operative", "Post-operative")


def _required(parent: ET.Element, tag: str) -> str:
    element = parent.find(tag)
    if element is None or element.text is None or not element.text.strip():
        raise ValueError(f"Missing required XML value: {tag}")
    return element.text.strip()


def _boolean(parent: ET.Element, tag: str) -> bool:
    value = _required(parent, tag).casefold()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{tag} must be true or false")
    return value in {"true", "1"}


def _list(parent: ET.Element, container: str, item_tag: str) -> list[str]:
    element = parent.find(container)
    if element is None:
        raise ValueError(f"Missing required XML section: {container}")
    values = [item.text.strip() for item in element.findall(item_tag) if item.text and item.text.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate value in {container}")
    return values


def _path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_pipeline_config(path: str | Path) -> dict:
    """Parse and semantically validate a pipeline XML file."""

    config_path = Path(path).resolve()
    try:
        root = ET.parse(config_path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"Unable to read pipeline XML {config_path}: {error}") from error
    if root.tag != "pipelineConfiguration" or root.get("version") != "1":
        raise ValueError("Pipeline XML must have a pipelineConfiguration root with version='1'")

    project_root = config_path.parent.parent
    preprocessing = root.find("preprocessing")
    model_folder = root.find("model")

    if preprocessing is None or model_folder is None:
        raise ValueError("Pipeline XML requires logging, preprocessing, and model sections")

    logging = root.find("logging")
    if logging is None:
        raise ValueError("Missing pipelineConfiguration/logging section")
    level = _required(logging, "level").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("Logging level must be DEBUG, INFO, WARNING, or ERROR")
    logging_config = {
        "directory": _path(project_root, _required(logging, "directory")),
        "level": level,
    }

    inputs_element = preprocessing.find("inputs")
    if inputs_element is None:
        raise ValueError("Missing preprocessing/inputs section")
    input_paths = [_path(project_root, item.text.strip()) for item in inputs_element.findall("path") if item.text and item.text.strip()]
    if not input_paths:
        raise ValueError("Preprocessing requires at least one input path")
    selection = preprocessing.find("parameterSelection")
    descriptions = preprocessing.find("parameterDescriptions")
    header_descriptions = preprocessing.find("csvHeaderDescriptions")
    plotting = preprocessing.find("plotting")
    relevant_parameters = preprocessing.find("relevantParameters")
    if any(section is None for section in (selection, descriptions, header_descriptions, plotting, relevant_parameters)):
        raise ValueError("Preprocessing requires relevantParameters, parameterSelection, parameterDescriptions, csvHeaderDescriptions, and plotting sections")
    parameter_map = load_parameter_map_element(relevant_parameters)
    phases = _list(selection, "phases", "phase")
    if not phases or any(phase not in PHASES for phase in phases):
        raise ValueError("Parameter-selection phases must include Pre-operative, During-operative, and/or Post-operative")
    parameter_selection = {
        "run": _boolean(selection, "run"),
        "phases": phases,
        "include_ids": _list(selection, "includeIds", "id"),
        "exclude_ids": _list(selection, "excludeIds", "id"),
    }
    bins = int(_required(plotting, "bins"))
    plot_phase = _required(plotting, "phase")
    if bins < 1 or plot_phase not in {"all", "preoperative", "duringoperative", "postoperative"}:
        raise ValueError("Preprocessing plotting requires positive bins and a supported phase")
    missingness_element = plotting.find("missingnessByPatient")
    if missingness_element is None:
        raise ValueError("Missing preprocessing/plotting/missingnessByPatient section")
    missingness_output = _path(project_root, _required(missingness_element, "output"))
    missingness_by_patient = {
        "run": _boolean(missingness_element, "run"),
        "output": missingness_output,
    }
    theme = _required(plotting, "theme")
    if theme not in {"light", "dark"}:
        raise ValueError("Preprocessing plotting theme must be light or dark")
    patient_timeline_element = plotting.find("patientTimeline")
    if patient_timeline_element is None:
        raise ValueError("Missing preprocessing/plotting/patientTimeline section")
    patient_timeline = {
        "run": _boolean(patient_timeline_element, "run"),
        "patient_ids": _list(patient_timeline_element, "patientIds", "id"),
        "compare_all_patients": _boolean(patient_timeline_element, "compareAllPatients"),
        "include_study_plots": _boolean(patient_timeline_element, "includeStudyPlots"),
        "layout": DashboardLayout.from_xml(patient_timeline_element.find("layout")),
        "output": _path(project_root, _required(patient_timeline_element, "output")),
    }
    time_series_element = plotting.find("timeSeries")
    if time_series_element is None:
        raise ValueError("Missing preprocessing/plotting/timeSeries section")
    group_by = _required(time_series_element, "groupBy")
    if group_by not in {"parameter", "class"}:
        raise ValueError("Time-series groupBy must be parameter or class")
    time_series_plotting = {
        "run": _boolean(time_series_element, "run"),
        "group_by": group_by,
        "parameter_ids": _list(time_series_element, "parameterIds", "id"),
        "class_names": _list(time_series_element, "classNames", "name"),
        "patient_ids": _list(time_series_element, "patientIds", "id"),
        "output_directory": _path(project_root, _required(time_series_element, "outputDirectory")),
    }
    preprocessing_run = _boolean(preprocessing, "run")
    if time_series_plotting["run"] and preprocessing_run and not parameter_map.get("export_time_series", False):
        raise ValueError("Time-series plotting with preprocessing requires export_time_series=true")
    preprocessing_config = {
        "run": preprocessing_run,
        "inputs": input_paths,
        "output_directory": _path(project_root, _required(preprocessing, "outputDirectory")),
        "parameter_map": parameter_map,
        "parameter_selection": parameter_selection,
        "parameter_descriptions": {
            "run": _boolean(descriptions, "run"),
            "output": _path(project_root, _required(descriptions, "output")),
        },
        "csv_header_descriptions": {
            "run": _boolean(header_descriptions, "run"),
            "inventory_output": _path(project_root, _required(header_descriptions, "inventoryOutput")),
            "summary_output": _path(project_root, _required(header_descriptions, "summaryOutput")),
        },
        "plotting": {
            "run": _boolean(plotting, "run"),
            "distributions": _boolean(plotting, "distributions"),
            "phase": plot_phase,
            "bins": bins,
            "theme": theme,
            "missingness_by_patient": missingness_by_patient,
            "time_series": time_series_plotting,
            "patient_timeline": patient_timeline,
        },
    }

    active_model, active_config = parse_active_model(model_folder, project_root)
    model_config = {
        "active_model": active_model,
        "models": {active_model: active_config.as_dict()},
        "active_config": active_config,
    }
    return {
        "project_root": project_root,
        "logging": logging_config,
        "preprocessing": preprocessing_config,
        "model": model_config,
    }


def select_parameter_map(
    parameter_map: dict,
    *,
    phases: list[str],
    include_ids: list[str],
    exclude_ids: list[str],
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
        raise ValueError(f"Parameter IDs cannot be both included and excluded: {', '.join(sorted(overlap))}")
    selected = [
        item for item in all_parameters
        if item.get("phase") in phases
        and (not include_ids or item["id"] in include_ids)
        and item["id"] not in exclude_ids
    ]
    if include_ids and {item["id"] for item in selected} != set(include_ids):
        raise ValueError("Every included parameter ID must belong to a selected phase")
    result = dict(parameter_map)
    result["parameters"] = selected
    if "postoperative_measurements" in result:
        postoperative_ids = {item["id"] for item in selected if item.get("phase") == "Post-operative"}
        result["postoperative_measurements"] = [
            item for item in result["postoperative_measurements"]
            if item.get("id") in postoperative_ids
        ]
    return result


def write_parameter_reference(parameter_map: dict, path: str | Path) -> None:
    """Write configured parameter descriptions for review, without cohort data."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Parameter reference", "",
        "Generated from the configured parameter map. It contains definitions, not patient data.", "",
        "| ID | Phase | Parameter | Description | Source fields | Type | Unit |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in parameter_map.get("parameters", []):
        source_fields = ", ".join(item.get("source_fields", [])) or "—"
        description = item.get("extraction_definition", "") or item.get("description", "") or "—"
        cells = [
            item.get("id", ""), item.get("phase", ""),
            item.get("clinical_parameter", item.get("name", "")),
            description, source_fields,
            item.get("parameter_type", ""), str(item.get("unit") or "—"),
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in cells) + " |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
