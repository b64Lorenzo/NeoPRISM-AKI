"""Execute preprocessing and model stages declared by the XML run file."""

from __future__ import annotations

from pathlib import Path

from src.logging.logger import Logger, LogLevel
from src.models.runner import get_model_runner
from src.pipeline.config import PipelineConfig
from src.preprocessing.csv_header_descriptions.inventory import write_csv_header_descriptions
from src.preprocessing.parameter_descriptions.reference import write_parameter_reference
from src.preprocessing.patient_pipeline import analyze_inputs
from src.preprocessing.plotting.distributions.plots import generate_parameter_distributions
from src.preprocessing.plotting.patient_timeline.patient_timeline import PatientTimelineDashboard
from src.preprocessing.plotting.time_series.time_series_plots import TimeSeriesPlotter


def _study_plot_artifacts(preprocessing: dict, plots: dict) -> list[Path]:
    candidates = [
        preprocessing["output_directory"] / "preoperative_heatmap.html",
        preprocessing["output_directory"] / "postoperative_heatmap.html",
        plots["missingness_by_patient"]["output"],
    ]
    for directory_name in ("missingness_heatmaps", "parameter_distributions"):
        directory = preprocessing["output_directory"] / directory_name
        if directory.is_dir():
            candidates.extend(directory.rglob("*.html"))
    time_series_dir = plots["time_series"]["output_directory"]
    if time_series_dir.is_dir():
        candidates.extend(time_series_dir.rglob("*.html"))
    unique = {path.resolve(): path.resolve() for path in candidates if path.is_file()}
    return [unique[path] for path in sorted(unique, key=lambda item: str(item).casefold())]



def _run_time_series_plot(
    preprocessing: dict, selected_map: dict, logger: Logger, completed: list[str]
) -> None:
    plots = preprocessing["plotting"]
    time_series = plots.get("time_series", {})
    if not time_series.get("run"):
        return

    logger.write(LogLevel.DEBUG, "preprocessing", f"starting time-series plot generation with config: {time_series}")
    result = TimeSeriesPlotter(selected_map).generate(
        preprocessing["output_directory"] / "time_series.csv",
        time_series["output_directory"],
        parameter_ids=time_series["parameter_ids"],
        class_names=time_series["class_names"],
        patient_ids=time_series["patient_ids"],
        group_by=time_series["group_by"],
        theme=plots["theme"],
    )
    logger.write(
        LogLevel.INFO,
        "time-series plots",
        f"plots={result['plot_count']}; parameters={result['parameter_count']}; "
        f"patients={result['patient_count']}; observations={result['observation_count']}",
    )
    completed.append("time-series plots")


def _run_patient_timeline(
    preprocessing: dict, selected_map: dict, logger: Logger, completed: list[str]
) -> None:
    plots = preprocessing["plotting"]
    patient_timeline = plots.get("patient_timeline", {})
    if not patient_timeline.get("run"):
        return

    logger.write(LogLevel.DEBUG, "preprocessing", f"starting patient timeline dashboard generation with config: {patient_timeline}")
    result = PatientTimelineDashboard(selected_map).generate(
        [preprocessing["input_directory"]],
        patient_timeline["output"],
        patient_ids=patient_timeline["patient_ids"],
        compare_all_patients=patient_timeline["compare_all_patients"],
        related_plots=(
            _study_plot_artifacts(preprocessing, plots)
            if patient_timeline["include_study_plots"] else []
        ),
        theme=plots["theme"],
        exclude_output_dir=preprocessing["output_directory"],
        layout=patient_timeline["layout"],
    )
    logger.write(
        LogLevel.INFO,
        "preprocessing",
        f"Patient timeline dashboard: patients={result['patient_count']}; events={result['event_count']}; "
        f"data_points={result['data_point_count']}",
    )
    completed.append("patient timeline dashboard")


def _run_plotting_stages(
    preprocessing: dict, selected_map: dict, logger: Logger, completed: list[str]
) -> None:
    plots = preprocessing["plotting"]
    distributions = plots.get("distributions", {})
    if plots["run"] and distributions.get("run"):
        logger.write(LogLevel.DEBUG, "preprocessing", f"starting distribution plot generation with config: {distributions}")
        count = generate_parameter_distributions(
            preprocessing["output_directory"],
            selected_map,
            bins=distributions["bins"],
            phase=distributions["phase"],
            parameter_ids=distributions["parameter_ids"],
            class_names=distributions["class_names"],
            theme=plots["theme"],
        )
        logger.write(LogLevel.INFO, "preprocessing", f"distribution plots generated; parameters={count}")
        completed.append("preprocessing plots")

    # Time-series plots can consume an existing CSV without rerunning extraction.
    _run_time_series_plot(preprocessing, selected_map, logger, completed)
    _run_patient_timeline(preprocessing, selected_map, logger, completed)


def _run_preprocessing_stages(
    preprocessing: dict,
    selected_map: dict,
    parameter_map: dict,
    logger: Logger,
    completed: list[str],
) -> list[str]:
    descriptions = preprocessing["parameter_descriptions"]
    try:
        if descriptions["run"]:
            logger.write(LogLevel.DEBUG, "preprocessing", f"starting parameter reference generation with config: {descriptions}")
            write_parameter_reference(selected_map, descriptions["output"])
            completed.append("parameter reference")
    except Exception as error:
        logger.write(LogLevel.WARNING, "preprocessing", f"error occurred while generating parameter reference: {error}")

    header_descriptions = preprocessing["csv_header_descriptions"]
    try:
        if header_descriptions["run"]:
            logger.write(LogLevel.DEBUG, "preprocessing", f"starting CSV header description generation with config: {header_descriptions}")
            counts = write_csv_header_descriptions(
                [preprocessing["input_directory"]],
                preprocessing["output_directory"],
                header_descriptions["inventory_output"],
                header_descriptions["summary_output"],
                parameter_map=parameter_map,
            )
            logger.write(LogLevel.INFO, "preprocessing", f"header summary complete; files={counts['files']}; unique_headers={counts['unique_headers']}")
            completed.append("CSV header descriptions")
    except Exception as error:
        logger.write(LogLevel.WARNING, "preprocessing", f"error occurred while generating CSV header descriptions: {error}")

    plots = preprocessing["plotting"]
    logger.write(LogLevel.INFO, "preprocessing", "starting patient extraction")
    patients = analyze_inputs(
        [preprocessing["input_directory"]],
        preprocessing["output_directory"],
        parameter_map,
        logger,
        generate_plots=plots["run"],
        generate_patient_missingness=(
            plots["run"] and plots["missingness_by_patient"]["run"]
        ),
        patient_missingness_output=plots["missingness_by_patient"]["output"],
        patient_missingness_selection={
            key: plots["missingness_by_patient"][key]
            for key in ("phase", "parameter_ids", "class_names")
        },
        plot_theme=plots["theme"],
    )
    logger.write(LogLevel.INFO, "preprocessing", f"patient extraction complete; patients={len(patients)}")
    completed.append("preprocessing")
    try:
        _run_plotting_stages(preprocessing, selected_map, logger, completed)
    except Exception as error:
        logger.write(LogLevel.WARNING, "preprocessing", f"error occurred while running plotting stages: {error}")
    return completed


def _run_existing_time_series_plot(
    preprocessing: dict, selected_map: dict, logger: Logger, completed: list[str]
) -> None:
    """Allow plot-only runs against an existing time_series.csv."""
    _run_time_series_plot(preprocessing, selected_map, logger, completed)

def run_pipeline(config_path: str | Path) -> str:
    configuration = PipelineConfig.load(config_path)
    config = configuration.as_dict()
    level = LogLevel[config["logging"]["level"]]
    log_directory = config["logging"]["directory"]
    completed = []

    with Logger(minimum_level=level, log_root=log_directory) as logger:
        logger.write(LogLevel.INFO, "cli", "starting XML-configured pipeline")
        preprocessing = configuration.preprocessing
        if preprocessing is not None:
            try:
                preprocessing_dict = preprocessing.as_dict()
                selected_map = preprocessing.parameters.selected_map()
                if preprocessing.run:
                    logger.write(LogLevel.INFO, "preprocessing", "starting preprocessing")
                    completed = _run_preprocessing_stages(
                        preprocessing_dict,
                        selected_map,
                        preprocessing.parameters.parameter_map,
                        logger,
                        completed,
                    )
                else:
                    _run_existing_time_series_plot(
                        preprocessing_dict, selected_map, logger, completed
                    )
            except Exception as error:
                logger.write(LogLevel.ERROR, "preprocessing", f"error occurred while running preprocessing stages: {error}")
                raise

        logger.write(LogLevel.INFO, "model", "starting model stages")
        for model_instance in configuration.models.models:
            if not model_instance.run:
                continue
            try:
                logger.write(
                    LogLevel.INFO,
                    "model",
                    f"starting model {model_instance.id} ({model_instance.type})",
                )
                model_runner = get_model_runner(model_instance.config)
                model_stages = model_runner(model_instance.config, logger)
                completed.extend(f"{model_instance.id}: {stage}" for stage in model_stages)
            except Exception as error:
                logger.write(
                    LogLevel.ERROR,
                    "model",
                    f"model {model_instance.id} failed ({type(error).__name__}): {error}",
                )
                raise

        summary = ", ".join(completed) if completed else "no stages enabled"
        logger.write(LogLevel.INFO, "cli", f"XML pipeline complete: {summary}")
    return summary
