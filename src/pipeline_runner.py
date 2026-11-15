"""Execute preprocessing and model stages declared by the XML run file."""

from __future__ import annotations

import json
from pathlib import Path

from src.csv_header_descriptions import write_csv_header_descriptions
from src.logger import LogLevel, Logger
from src.model_config import PerceptronModelConfig
from src.parameter_distributions import generate_parameter_distributions
from src.pipeline_config import load_pipeline_config, select_parameter_map, write_parameter_reference
from src.patient_pipeline import analyze_inputs
from src.patient_timeline import PatientTimelineDashboard
from src.perceptron import plot_confusion_matrix, predict_model, train_model
from src.time_series_plots import TimeSeriesPlotter
from src.training_io import join_patient_vectors, read_csv_rows, write_csv_rows


def _study_plot_artifacts(preprocessing: dict, plots: dict, model_config: dict) -> list[Path]:
    candidates = [
        preprocessing["output_directory"] / "preoperative_heatmap.html",
        preprocessing["output_directory"] / "postoperative_heatmap.html",
        plots["missingness_by_patient"]["output"],
    ]
    model_plot = model_config.get("plotting", {})
    if model_plot.get("run") and model_plot.get("output"):
        candidates.append(model_plot["output"])
    for directory_name in ("missingness_heatmaps", "parameter_distributions"):
        directory = preprocessing["output_directory"] / directory_name
        if directory.is_dir():
            candidates.extend(directory.rglob("*.html"))
    time_series_dir = plots["time_series"]["output_directory"]
    if time_series_dir.is_dir():
        candidates.extend(time_series_dir.rglob("*.html"))
    unique = {path.resolve(): path.resolve() for path in candidates if path.is_file()}
    return [unique[path] for path in sorted(unique, key=lambda item: str(item).casefold())]


def _configured_selection(preprocessing: dict) -> dict | None:
    selection = preprocessing["parameter_selection"]
    if not selection["run"]:
        return None
    return {
        "phases": selection["phases"],
        "include_ids": selection["include_ids"],
        "exclude_ids": selection["exclude_ids"],
    }


def run_pipeline(config_path: str | Path) -> str:
    config = load_pipeline_config(config_path)
    preprocessing = config["preprocessing"]
    model_config = config["model"]["active_config"]
    if not isinstance(model_config, PerceptronModelConfig):
        raise ValueError(f"No runner is registered for model: {model_config.xml_name}")
    perceptron_config = model_config.as_dict()
    selection = _configured_selection(preprocessing)
    parameter_map = preprocessing["parameter_map"]
    selected_map = (
        select_parameter_map(parameter_map, **selection)
        if selection is not None else parameter_map
    )
    level = LogLevel[config["logging"]["level"]]
    log_directory = config["logging"]["directory"]
    completed = []

    with Logger(minimum_level=level, log_root=log_directory) as logger:
        logger.write(LogLevel.INFO, "cli", "starting XML-configured pipeline")
        descriptions = preprocessing["parameter_descriptions"]
        if descriptions["run"]:
            write_parameter_reference(selected_map, descriptions["output"])
            completed.append("parameter reference")

        header_descriptions = preprocessing["csv_header_descriptions"]
        if header_descriptions["run"]:
            counts = write_csv_header_descriptions(
                preprocessing["inputs"],
                preprocessing["output_directory"],
                header_descriptions["inventory_output"],
                header_descriptions["summary_output"],
                parameter_map=parameter_map,
            )
            logger.write(LogLevel.INFO, "csv headers", f"header summary complete; files={counts['files']}; unique_headers={counts['unique_headers']}")
            completed.append("CSV header descriptions")

        plots = preprocessing["plotting"]
        if preprocessing["run"]:
            analyze_inputs(
                preprocessing["inputs"],
                preprocessing["output_directory"],
                selected_map,
                logger,
                generate_plots=plots["run"],
                generate_patient_missingness=(
                    plots["run"] and plots["missingness_by_patient"]["run"]
                ),
                patient_missingness_output=plots["missingness_by_patient"]["output"],
                plot_theme=plots["theme"],
            )
            completed.append("preprocessing")
        if plots["run"] and plots["distributions"]:
            count = generate_parameter_distributions(
                preprocessing["output_directory"],
                selected_map,
                bins=plots["bins"],
                phase=plots["phase"],
                theme=plots["theme"],
            )
            logger.write(LogLevel.INFO, "plotting", f"distribution plots generated; parameters={count}")
            completed.append("preprocessing plots")

        time_series_plots = plots["time_series"]
        if time_series_plots["run"]:
            result = TimeSeriesPlotter(selected_map).generate(
                preprocessing["output_directory"] / "time_series.csv",
                time_series_plots["output_directory"],
                parameter_ids=time_series_plots["parameter_ids"],
                class_names=time_series_plots["class_names"],
                patient_ids=time_series_plots["patient_ids"],
                group_by=time_series_plots["group_by"],
                theme=plots["theme"],
            )
            logger.write(
                LogLevel.INFO,
                "time-series plots",
                f"plots={result['plot_count']}; parameters={result['parameter_count']}; "
                f"patients={result['patient_count']}; observations={result['observation_count']}",
            )
            completed.append("time-series plots")

        if perceptron_config["run"]:
            if perceptron_config["mode"] == "train-validation":
                settings = perceptron_config["training_validation"]
                if preprocessing["run"]:
                    vectors_path = preprocessing["output_directory"] / "patient_vectors.csv"
                    outcomes_path = preprocessing["output_directory"] / "patient_outcomes.csv"
                else:
                    vectors_path = settings["vectors"]
                    outcomes_path = settings["outcomes"]
                _, vector_rows = read_csv_rows(vectors_path)
                _, outcome_rows = read_csv_rows(outcomes_path)
                rows = join_patient_vectors(vector_rows, outcome_rows)
                features = [
                    item for item in selected_map["parameters"]
                    if item.get("phase") == "Pre-operative"
                ]
                if not features:
                    raise ValueError("Perceptron training needs at least one selected pre-operative parameter")
                model_result = train_model(
                    rows,
                    features,
                    omitted_parameters=settings["omit"],
                    train_fraction=settings["train_fraction"],
                    epochs=settings["epochs"],
                    seed=settings["seed"],
                    include_sets=True,
                )
                training_rows = model_result.pop("_training_set")
                validation_rows = model_result.pop("_validation_set")
                feature_ids = model_result["feature_ids"]
                write_csv_rows(
                    settings["training_set"], training_rows,
                    ["patient_id", *feature_ids, "target"],
                )
                write_csv_rows(
                    settings["validation_set"], validation_rows,
                    ["patient_id", *feature_ids, "prediction"],
                )
                settings["model_output"].parent.mkdir(parents=True, exist_ok=True)
                settings["model_output"].write_text(
                    json.dumps(model_result, indent=2) + "\n", encoding="utf-8"
                )
                if perceptron_config["plotting"]["run"]:
                    plot_confusion_matrix(
                        model_result, perceptron_config["plotting"]["output"], theme=plots["theme"],
                    )
                    completed.append("perceptron plot")
                logger.write(
                    LogLevel.INFO,
                    "perceptron",
                    f"training complete; training_rows={model_result['training_size']}; "
                    f"validation_rows={model_result['validation_size']}; "
                    f"misclassified={model_result['misclassified']}; "
                    f"mislabeling_rate={model_result['mislabeling_rate']:.6f}",
                )
                completed.append("perceptron training/validation")
            else:
                settings = perceptron_config["novel_data"]
                vectors_path = (
                    preprocessing["output_directory"] / "patient_vectors.csv"
                    if preprocessing["run"] else settings["vectors"]
                )
                fields, vector_rows = read_csv_rows(vectors_path)
                if "target" in fields:
                    raise ValueError("Novel-data input must contain features only, without target labels")
                model = json.loads(settings["model_input"].read_text(encoding="utf-8"))
                predictions = predict_model(model, vector_rows)
                write_csv_rows(
                    settings["predictions"], predictions,
                    ["patient_id", *model["feature_ids"], "prediction"],
                )
                logger.write(
                    LogLevel.INFO, "perceptron",
                    f"novel-data predictions complete; rows={len(predictions)}",
                )
                completed.append("novel-data predictions")

        patient_timeline = plots["patient_timeline"]
        if patient_timeline["run"]:
            result = PatientTimelineDashboard(selected_map).generate(
                preprocessing["inputs"],
                patient_timeline["output"],
                patient_ids=patient_timeline["patient_ids"],
                compare_all_patients=patient_timeline["compare_all_patients"],
                related_plots=(
                    _study_plot_artifacts(preprocessing, plots, perceptron_config)
                    if patient_timeline["include_study_plots"] else []
                ),
                theme=plots["theme"],
                exclude_output_dir=preprocessing["output_directory"],
                layout=patient_timeline["layout"],
            )
            logger.write(
                LogLevel.INFO,
                "patient timeline",
                f"patients={result['patient_count']}; events={result['event_count']}; "
                f"data_points={result['data_point_count']}",
            )
            completed.append("patient timeline dashboard")

    return ", ".join(completed) if completed else "no stages enabled"
