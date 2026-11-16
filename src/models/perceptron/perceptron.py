"""Small online Rosenblatt perceptron and patient-level holdout evaluation."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.plotlib.plotly_output import write_plotly_html

NUMERIC_TYPES = {"numeric", "derived_numeric", "time_series_numeric", "derived_duration"}
MISSING_VALUES = {"", "na", "n/a", "none", "null"}


def _label(value: object) -> int | None:
    text = "" if value is None else str(value).strip()
    if text.casefold() in MISSING_VALUES:
        return None
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError("Outcome targets must be numeric; expected -1 or 1") from error
    if number not in (-1, 1):
        raise ValueError("Outcome targets must be exactly -1 (failure) or 1 (success)")
    return int(number)


def _value(row: Mapping[str, object], parameter_id: str) -> str:
    value = row.get(parameter_id)
    text = "" if value is None else str(value).strip()
    if text.casefold() in MISSING_VALUES:
        raise ValueError(f"Missing input value for parameter: {parameter_id}")
    return text


def _fit_encoder(rows: Sequence[Mapping[str, object]], parameters: Sequence[Mapping]):
    names: list[str] = []
    numeric: dict[str, dict[str, float]] = {}
    categorical: dict[str, list[str]] = {}
    for parameter in parameters:
        parameter_id = str(parameter["id"])
        if parameter.get("parameter_type") in NUMERIC_TYPES:
            try:
                values = [float(_value(row, parameter_id)) for row in rows]
            except ValueError as error:
                raise ValueError(f"Missing or invalid numeric input for parameter: {parameter_id}") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite numeric input for parameter: {parameter_id}")
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            numeric[parameter_id] = {"mean": mean, "scale": math.sqrt(variance) or 1.0}
            names.append(parameter_id)
        else:
            levels = sorted({_value(row, parameter_id) for row in rows}, key=str.casefold)
            categorical[parameter_id] = levels
            names.extend(f"{parameter_id}={level}" for level in levels)
    return names, numeric, categorical


def _encode_rows(rows, parameters, numeric, categorical):
    encoded = []
    for row in rows:
        vector = []
        for parameter in parameters:
            parameter_id = str(parameter["id"])
            text = _value(row, parameter_id)
            if parameter_id in numeric:
                try:
                    value = float(text)
                except ValueError as error:
                    raise ValueError(f"Invalid numeric input for parameter: {parameter_id}") from error
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite numeric input for parameter: {parameter_id}")
                stats = numeric[parameter_id]
                scaled = (value - stats["mean"]) / stats["scale"]
                if not math.isfinite(scaled):
                    raise ValueError(f"Non-finite encoded numeric input for parameter: {parameter_id}")
                vector.append(scaled)
            else:
                vector.extend(float(text == level) for level in categorical[parameter_id])
        encoded.append(vector)
    return encoded


class Perceptron:
    """Online Rosenblatt learning, with no bias term."""

    def __init__(self, feature_count: int) -> None:
        self.weights = [0.0] * feature_count
        self.updates = 0

    def predict(self, features: Sequence[float]) -> int:
        return 1 if sum(weight * value for weight, value in zip(self.weights, features)) >= 0 else -1

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int], epochs: int):
        completed = 0
        converged = False
        for epoch in range(epochs):
            errors = 0
            for features, label in zip(rows, labels):
                potential = sum(weight * value for weight, value in zip(self.weights, features))
                if label * potential <= 0:
                    size = len(features) or 1
                    self.weights = [
                        weight + label * value / size
                        for weight, value in zip(self.weights, features)
                    ]
                    self.updates += 1
                    errors += 1
            completed = epoch + 1
            if errors == 0:
                converged = True
                break
        return completed, converged


def train_model(
    rows: Sequence[Mapping[str, object]],
    parameters: Sequence[Mapping],
    *,
    omitted_parameters: Sequence[str] = (),
    train_fraction: float = 0.8,
    epochs: int = 100,
    seed: int = 0,
    include_sets: bool = False,
) -> dict:
    """Train on labeled rows; classify the holdout before reading its labels."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    if not rows:
        raise ValueError("No patient vectors were provided")
    all_ids = {str(parameter["id"]) for parameter in parameters}
    omitted = list(dict.fromkeys(str(item).strip() for item in omitted_parameters if str(item).strip()))
    unknown = sorted(set(omitted) - all_ids)
    if unknown:
        raise ValueError(f"Unknown omitted parameter: {', '.join(unknown)}")
    available = [
        parameter for parameter in parameters
        if str(parameter["id"]) not in omitted and str(parameter["id"]) in rows[0]
    ]
    if not available:
        raise ValueError("No input parameters remain; check the parameter map and omissions")
    feature_ids = [str(parameter["id"]) for parameter in available]

    labeled_rows, labels = [], []
    excluded_rows = 0
    for row in rows:
        target = _label(row.get("target"))
        missing_feature = any(
            not str(row.get(key, "")).strip()
            or str(row.get(key, "")).strip().casefold() in MISSING_VALUES
            for key in feature_ids
        )
        if target is None or missing_feature:
            excluded_rows += 1
            continue
        labeled_rows.append(row)
        labels.append(target)
    if len(labeled_rows) < 3:
        raise ValueError(
            "At least three labeled complete rows are required for training and validation; "
            f"found {len(labeled_rows)} after excluding {excluded_rows} row(s)"
        )
    if len(set(labels)) != 2:
        raise ValueError("Training data must contain both target classes: -1 and 1")

    indices = list(range(len(labeled_rows)))
    random.Random(seed).shuffle(indices)
    train_size = int(len(indices) * train_fraction)
    if train_size < 2 or train_size >= len(indices):
        raise ValueError("train_fraction must leave at least two training rows and one validation row")
    train_indices, validation_indices = indices[:train_size], indices[train_size:]
    for missing_label in (-1, 1):
        if missing_label in {labels[i] for i in train_indices}:
            continue
        position = next(
            pos for pos in range(len(train_indices) - 1, -1, -1)
            if labels[train_indices[pos]] != missing_label
        )
        replacement = next(i for i in validation_indices if labels[i] == missing_label)
        validation_indices.append(train_indices[position])
        train_indices[position] = replacement
        validation_indices.remove(replacement)

    train_rows = [labeled_rows[i] for i in train_indices]
    validation_rows = [labeled_rows[i] for i in validation_indices]
    train_targets = [labels[i] for i in train_indices]
    validation_targets = [labels[i] for i in validation_indices]
    names, numeric, categorical = _fit_encoder(train_rows, available)
    train_inputs = _encode_rows(train_rows, available, numeric, categorical)
    validation_inputs = _encode_rows(validation_rows, available, numeric, categorical)
    classifier = Perceptron(len(names))
    epochs_trained, converged = classifier.fit(train_inputs, train_targets, epochs)

    # Predict using features only. Ground-truth labels are compared only afterward.
    predictions = [classifier.predict(vector) for vector in validation_inputs]
    counts = {
        "failure_predicted_failure": 0,
        "failure_predicted_success": 0,
        "success_predicted_failure": 0,
        "success_predicted_success": 0,
    }
    for actual, predicted in zip(validation_targets, predictions):
        actual_name = "failure" if actual == -1 else "success"
        predicted_name = "failure" if predicted == -1 else "success"
        counts[f"{actual_name}_predicted_{predicted_name}"] += 1
    misclassified = counts["failure_predicted_success"] + counts["success_predicted_failure"]
    correct = len(validation_targets) - misclassified

    result = {
        "algorithm": "Rosenblatt perceptron",
        "target_values": [-1, 1],
        "excluded_rows": excluded_rows,
        "feature_ids": feature_ids,
        "omitted_parameters": omitted,
        "feature_names": names,
        "numeric_transform": numeric,
        "categorical_values": categorical,
        "weights": classifier.weights,
        "training_size": len(train_indices),
        "validation_size": len(validation_indices),
        "train_fraction": train_fraction,
        "seed": seed,
        "epochs": epochs_trained,
        "updates": classifier.updates,
        "converged": converged,
        "misclassified": misclassified,
        "correctly_classified": correct,
        "mislabeling_rate": misclassified / len(validation_targets),
        "generalization_error": misclassified / len(validation_targets),
        "confusion": counts,
    }
    if include_sets:
        result["_training_set"] = [
            {
                "patient_id": row.get("patient_id", ""),
                **{key: row[key] for key in feature_ids},
                "target": label,
            }
            for row, label in zip(train_rows, train_targets)
        ]
        result["_validation_set"] = [
            {
                "patient_id": row.get("patient_id", ""),
                **{key: row[key] for key in feature_ids},
                "prediction": prediction,
            }
            for row, prediction in zip(validation_rows, predictions)
        ]
    return result


def predict_model(model: Mapping, rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """Apply a saved perceptron and its training-only encoder to novel vectors."""
    feature_ids = [str(item) for item in model.get("feature_ids", [])]
    feature_names = [str(item) for item in model.get("feature_names", [])]
    numeric = model.get("numeric_transform", {})
    categorical = model.get("categorical_values", {})
    weights = model.get("weights", [])
    if not isinstance(numeric, Mapping):
        raise ValueError("Saved model numeric transform must be an object")
    numeric_stats = {}
    for parameter_id, stats in numeric.items():
        try:
            mean, scale = (float(stats[key]) for key in ("mean", "scale"))
            valid = not any(isinstance(stats[key], bool) for key in ("mean", "scale"))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Saved model numeric transform is invalid for parameter: {parameter_id}") from error
        if not valid or not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Saved model numeric transform needs a finite mean and positive finite scale: {parameter_id}")
        numeric_stats[parameter_id] = {"mean": mean, "scale": scale}
    numeric = numeric_stats
    if not feature_ids or len(set(feature_ids)) != len(feature_ids):
        raise ValueError("Saved model must contain unique feature IDs")
    parameters = [
        {"id": item, "parameter_type": "numeric" if item in numeric else "categorical"}
        for item in feature_ids
    ]
    encoded_names = []
    for parameter in parameters:
        parameter_id = parameter["id"]
        if parameter_id in numeric:
            encoded_names.append(parameter_id)
        elif parameter_id in categorical:
            encoded_names.extend(f"{parameter_id}={value}" for value in categorical[parameter_id])
        else:
            raise ValueError(f"Saved model has no encoder for parameter: {parameter_id}")
    if encoded_names != feature_names or len(weights) != len(feature_names):
        raise ValueError("Saved model feature names and weights are inconsistent")
    try:
        numeric_weights = [float(value) for value in weights]
    except (TypeError, ValueError) as error:
        raise ValueError("Saved model weights must be numeric") from error
    if not all(math.isfinite(value) for value in numeric_weights):
        raise ValueError("Saved model weights must be finite")

    classifier = Perceptron(len(numeric_weights))
    classifier.weights = numeric_weights
    predictions = []
    seen_ids = set()
    for row in rows:
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id or patient_id in seen_ids:
            raise ValueError("Novel vectors must have unique, non-empty patient_id values")
        if "target" in row:
            raise ValueError("Novel vectors must not contain postoperative target labels")
        seen_ids.add(patient_id)
        vector = _encode_rows([row], parameters, numeric, categorical)[0]
        predictions.append({
            "patient_id": patient_id,
            **{feature_id: row[feature_id] for feature_id in feature_ids},
            "prediction": classifier.predict(vector),
        })
    return predictions


def plot_confusion_matrix(model: Mapping, path: str | Path, theme: str = "light") -> None:
    """Write an interactive aggregate validation confusion matrix."""
    from src.plotlib.plotly_output import plotly_template

    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError("Perceptron plots require Plotly; install requirements.txt") from error

    confusion = model["confusion"]
    labels = ["Failure", "Success"]
    matrix = [
        [confusion["failure_predicted_failure"], confusion["failure_predicted_success"]],
        [confusion["success_predicted_failure"], confusion["success_predicted_success"]],
    ]
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                colorbar={"title": {"text": "Patient count"}},
                hovertemplate=(
                    "Actual: %{y}<br>Predicted: %{x}<br>Patient count: %{z}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title="Perceptron validation confusion matrix",
        template=plotly_template(theme),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=600,
        height=500,
        margin={"l": 90, "r": 50, "t": 80, "b": 80},
    )
    figure.update_yaxes(autorange="reversed")
    write_plotly_html(figure, path)
