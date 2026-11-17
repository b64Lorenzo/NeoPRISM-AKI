"""Perceptron workflows for the XML-configured pipeline."""

from __future__ import annotations

import json

from src.logging.logger import Logger, LogLevel
from src.models.perceptron.config import PerceptronModelConfig
from src.models.perceptron.perceptron import plot_confusion_matrix, predict_model, train_model
from src.models.training_validation.config import TrainingValidationConfig
from src.utils.table_io import join_patient_vectors, read_csv_rows, write_csv_rows


def train_and_save(settings: TrainingValidationConfig) -> dict:
    _, vectors = read_csv_rows(settings.vectors)
    _, outcomes = read_csv_rows(settings.outcomes)
    rows = join_patient_vectors(vectors, outcomes)
    features = [feature.as_parameter() for feature in settings.features]
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
    logger: Logger,
) -> list[str]:
    completed = []
    if config.mode == "train-validation":
        settings = config.training_validation
        model = train_and_save(settings)
        if config.plotting.run:
            plot_confusion_matrix(model, config.plotting.output)
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
    fields, rows = read_csv_rows(settings.vectors)
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
