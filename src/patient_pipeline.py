"""Extract pre-operative patient vectors and separately keyed outcome labels."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from collections.abc import Mapping

from src.logger import LogLevel, Logger
from src.patient_heatmaps import sample_patients, write_heatmap, write_parameter_heatmap
from src.pipeline import discover_csv_files, group_patient_files, load_parameter_map
from src.duringoperative import DuringoperativeProcessor
from src.postoperative import PostoperativeProcessor
from src.preoperative import PreoperativeProcessor
from src.time_series import TimeSeriesProcessor


def _format_target(value: int | None) -> str:
    return "NA" if value is None else str(value)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_inputs(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    parameter_map_path: str | Path | Mapping,
    logger: Logger,
    outcome_column: str | None = None,
    success_values: Iterable[str] | None = None,
    failure_values: Iterable[str] | None = None,
    parameter_selection: Mapping[str, object] | None = None,
    generate_plots: bool = True,
    generate_patient_missingness: bool | None = None,
    patient_missingness_output: str | Path | None = None,
    plot_theme: str = "light",
) -> list[dict]:
    """Write a feature-only vector CSV and a separate outcome-label CSV."""
    output = Path(output_dir)
    parameter_map = (
        parameter_map_path
        if isinstance(parameter_map_path, Mapping)
        else load_parameter_map(parameter_map_path)
    )
    if parameter_selection is not None:
        from src.pipeline_config import select_parameter_map
        parameter_map = select_parameter_map(parameter_map, **parameter_selection)
    output.mkdir(parents=True, exist_ok=True)
    files = discover_csv_files(inputs, output)
    if not files:
        raise ValueError("No CSV files found in the provided input paths")

    preprocessor = PreoperativeProcessor(parameter_map)
    during_processor = DuringoperativeProcessor(parameter_map)
    series_processor = TimeSeriesProcessor(parameter_map)
    outcome_processor = PostoperativeProcessor(
        parameter_map,
        outcome_column=outcome_column,
        success_values=success_values,
        failure_values=failure_values,
    )
    features = list(preprocessor.parameter_specs)
    postoperative_parameters = list(outcome_processor.measurement_specs)
    patient_files = group_patient_files(files)
    patients = []
    with ExitStack() as outputs:
        series_writer = None
        if series_processor.enabled:
            stream = outputs.enter_context((output / "time_series.csv").open("w", newline="", encoding="utf-8"))
            series_writer = csv.DictWriter(
                stream, fieldnames=["patient_id", "phase", "parameter_id", "t_seconds", "value"]
            )
            series_writer.writeheader()
        for index, paths in enumerate(patient_files, start=1):
            patient_id = f"patient_{index:03d}"
            preoperative = preprocessor.process(paths)
            duringoperative = during_processor.process(paths, preoperative)
            outcome = outcome_processor.process(paths, preoperative)
            if series_writer is not None:
                series_writer.writerows(
                    {"patient_id": patient_id, **reading}
                    for reading in series_processor.process(paths, preoperative)
                )
            patients.append({
                "patient_id": patient_id,
                "features": preoperative.features,
                "duringoperative_measurements": duringoperative,
                "has_start": preoperative.cutoff is not None,
                "has_end": preoperative.end_cutoff is not None,
                "missing_features": preoperative.missing_features,
                "postoperative_measurements": outcome.measurements,
                "target": outcome.target,
                "target_status": outcome.target_status,
                "label_basis": outcome.label_basis,
            })

    feature_ids = [parameter["id"] for parameter in features]
    _write_csv(
        output / "patient_vectors.csv",
        ["patient_id", *feature_ids],
        ({"patient_id": patient["patient_id"], **patient["features"]} for patient in patients),
    )
    _write_csv(
        output / "patient_outcomes.csv",
        ["patient_id", "target", "target_status", "label_basis"],
        ({
            "patient_id": patient["patient_id"],
            "target": _format_target(patient["target"]),
            "target_status": patient["target_status"],
            "label_basis": patient["label_basis"],
        } for patient in patients),
    )
    if during_processor.parameters:
        during_ids = [parameter["id"] for parameter in during_processor.parameters]
        _write_csv(
            output / "duringoperative_measurements.csv",
            ["patient_id", *during_ids],
            ({"patient_id": patient["patient_id"], **patient["duringoperative_measurements"]}
             for patient in patients),
        )
    postoperative_ids = [parameter["id"] for parameter in postoperative_parameters]
    _write_csv(
        output / "postoperative_measurements.csv",
        ["patient_id", *postoperative_ids],
        ({"patient_id": patient["patient_id"], **patient["postoperative_measurements"]} for patient in patients),
    )

    if generate_plots:
        categorical_ids = {
            parameter["id"] for parameter in features
            if parameter.get("parameter_type") == "categorical"
        }
        write_parameter_heatmap(
            output / "preoperative_heatmap.html", patients, features,
            value_key="features", title="Pre-operative patient vectors",
            categorical_ids=categorical_ids,
            theme=plot_theme,
        )
        write_parameter_heatmap(
            output / "postoperative_heatmap.html", patients, postoperative_parameters,
            value_key="postoperative_measurements", title="Post-operative measurements",
            theme=plot_theme,
        )

    if generate_patient_missingness is None:
        generate_patient_missingness = generate_plots
    if generate_patient_missingness:
        sample = sample_patients(patients)
        missing_matrix = [
            [1.0 if feature_id in patient["missing_features"] else 0.0 for feature_id in feature_ids]
            for patient in sample
        ]
        write_heatmap(
            Path(patient_missingness_output)
            if patient_missingness_output is not None
            else output / "preoperative_missingness_heatmap.html",
            missing_matrix,
            [patient["patient_id"] for patient in sample],
            feature_ids,
            "Pre-operative feature missingness by patient",
            "Missing (1) / present (0)",
            row_axis_label="Patient pseudonym",
            colorscale=[[0, "#2f855a"], [1, "#c53030"]],
            colorbar_tickvals=[0, 1],
            colorbar_ticktext=["Present", "Missing"],
            theme=plot_theme,
        )

    failures = sum(patient["target_status"] == "failure" for patient in patients)
    successes = sum(patient["target_status"] == "success" for patient in patients)
    unclassified = len(patients) - failures - successes
    missing_start = sum(not patient["has_start"] for patient in patients)
    missing_end = sum(not patient["has_end"] for patient in patients)
    report = [
        "# Patient vector extraction", "",
        f"- Input CSV files: {len(files)}",
        f"- Patients: {len(patients)}",
        f"- Pre-operative input features: {len(features)}",
        f"- Patients without a timestamped operation start: {missing_start}",
        f"- Patients without a matched operation end after start: {missing_end}",
        f"- Labeled failures (-1): {failures}",
        f"- Labeled successes (+1): {successes}",
        f"- Unclassified: {unclassified}", "",
        "`patient_vectors.csv` contains only patient_id and pre-operative features. `patient_outcomes.csv` contains labels derived from postoperative evidence and is joined by patient_id during validation. Post-operative values are never perceptron inputs.", "",
        "Outcome criteria are parameter-specific and provisional; missing or undefined criteria remain unclassified.",
    ]
    (output / "extraction_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.write(
        LogLevel.INFO, "patient_pipeline",
        f"analysis completed; input_files={len(files)}; patients={len(patients)}; failure={failures}; success={successes}; unclassified={unclassified}; features={len(features)}",
    )
    return [
        {"patient_id": patient["patient_id"], "target": patient["target"],
         "missing_count": len(patient["missing_features"]), "preoperative_fields": len(feature_ids)}
        for patient in patients
    ]
