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




def run_pipeline(config_path: str | Path) -> str:
    configuration = PipelineConfig.load(config_path)
    config = configuration.as_dict()
    preprocessing = config["preprocessing"]
    model_config = configuration.model.active_config
    model_runner = get_model_runner(model_config)
    model_settings = model_config.as_dict()
    parameter_map = configuration.preprocessing.parameters.parameter_map
    selected_map = configuration.preprocessing.parameters.selected_map()
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
                [preprocessing["input_directory"]],
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
                [preprocessing["input_directory"]],
                preprocessing["output_directory"],
                selected_map,
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
            completed.append("preprocessing")
        if plots["run"] and plots["distributions"]["run"]:
            count = generate_parameter_distributions(
                preprocessing["output_directory"],
                selected_map,
                bins=plots["distributions"]["bins"],
                phase=plots["distributions"]["phase"],
                parameter_ids=plots["distributions"]["parameter_ids"],
                class_names=plots["distributions"]["class_names"],
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

        completed.extend(model_runner(
            model_config, selected_map, logger,
            generated_directory=preprocessing["output_directory"] if preprocessing["run"] else None,
            theme=plots["theme"],
        ))

        patient_timeline = plots["patient_timeline"]
        if patient_timeline["run"]:
            result = PatientTimelineDashboard(selected_map).generate(
                [preprocessing["input_directory"]],
                patient_timeline["output"],
                patient_ids=patient_timeline["patient_ids"],
                compare_all_patients=patient_timeline["compare_all_patients"],
                related_plots=(
                    _study_plot_artifacts(preprocessing, plots, model_settings)
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

        summary = ", ".join(completed) if completed else "no stages enabled"
        logger.write(LogLevel.INFO, "cli", f"XML pipeline complete: {summary}")
    return summary
