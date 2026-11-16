"""Shared perceptron workflows for the direct CLI and XML pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from src.logging.logger import Logger, LogLevel
from src.models.perceptron.config import PerceptronModelConfig
from src.models.perceptron.perceptron import plot_confusion_matrix, predict_model, train_model
from src.models.training_validation.config import TrainingValidationConfig
from src.utils.table_io import join_patient_vectors, read_csv_rows, write_csv_rows


def train_and_save(settings: TrainingValidationConfig, parameter_map: dict) -> dict:
    _, vectors = read_csv_rows(settings.vectors)
    _, outcomes = read_csv_rows(settings.outcomes)
    rows = join_patient_vectors(vectors, outcomes)
    features = [
        item for item in parameter_map["parameters"] if item.get("phase") == "Pre-operative"
    ]
    if not features:
        raise ValueError("Perceptron training needs at least one selected pre-operative parameter")
    model = train_model(
        rows,
        features,
        omitted_parameters=settings.omit,
        train_fraction=settings.train_fraction,
        epochs=settings.epochs,
        seed=settings.seed,
        include_sets=True,
    )
    fields = ["patient_id", *model["feature_ids"]]
    write_csv_rows(settings.training_set, model.pop("_training_set"), [*fields, "target"])
    write_csv_rows(settings.validation_set, model.pop("_validation_set"), [*fields, "prediction"])
    settings.model_output.parent.mkdir(parents=True, exist_ok=True)
    settings.model_output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return model


def run_perceptron(
    config: PerceptronModelConfig,
    parameter_map: dict,
    logger: Logger,
    *,
    generated_directory: Path | None = None,
    theme: str = "light",
) -> list[str]:
    if not config.run:
        return []
    completed = []
    if config.mode == "train-validation":
        settings = config.training_validation
        if generated_directory is not None:
            settings = replace(
                settings,
                vectors=generated_directory / "patient_vectors.csv",
                outcomes=generated_directory / "patient_outcomes.csv",
            )
        model = train_and_save(settings, parameter_map)
        if config.plotting.run:
            plot_confusion_matrix(model, config.plotting.output, theme=theme)
            completed.append("perceptron plot")
        logger.write(
            LogLevel.INFO,
            "perceptron",
            f"training complete; training_rows={model['training_size']}; "
            f"validation_rows={model['validation_size']}; "
            f"misclassified={model['misclassified']}; mislabeling_rate={model['mislabeling_rate']:.6f}",
        )
        return [*completed, "perceptron training/validation"]
    settings = config.novel_data
    vectors = (
        generated_directory / "patient_vectors.csv"
        if generated_directory is not None
        else settings.vectors
    )
    fields, rows = read_csv_rows(vectors)
    if "target" in fields:
        raise ValueError("Novel-data input must contain features only, without target labels")
    model = json.loads(settings.model_input.read_text(encoding="utf-8"))
    predictions = predict_model(model, rows)
    write_csv_rows(
        settings.predictions, predictions, ["patient_id", *model["feature_ids"], "prediction"]
    )
    logger.write(
        LogLevel.INFO, "perceptron", f"novel-data predictions complete; rows={len(predictions)}"
    )
    return ["novel-data predictions"]
