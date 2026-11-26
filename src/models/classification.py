"""Patient-level orchestration shared by classification baselines."""

import csv
import json
from pathlib import Path

from src.models.contracts import TARGETS, is_forbidden_predictor
from src.models.data import Transformer, read_matrix, split_rows
from src.models.evaluation.report import save_classification_report
from src.models.rosenblatt_perceptron.core import train as train_rosenblatt
from src.models.rosenblatt_perceptron.execution.predict import (
    predict_csv as predict_rosenblatt,
)
from src.models.rosenblatt_perceptron.plots.charts import save_training_plots
from src.models.rosenblatt_perceptron.validation.report import save_report

MODEL_TYPES = ("rosenblatt_perceptron", "logistic_regression", "multilayer_perceptron")


def _probabilistic_model(kind, width, classes, seed, l2, hidden_layers, activation):
    if kind == "logistic_regression":
        from src.models.logistic_regression import LogisticRegression

        return LogisticRegression(l2=l2, seed=seed)
    if l2 != 0:
        raise ValueError("l2 is currently supported only by logistic regression")
    from src.models.neural_network import MultiLayerPerceptron

    shape = [width, *hidden_layers, len(classes)]
    return MultiLayerPerceptron(
        shape, [activation] * len(hidden_layers) + ["softmax"], seed=seed
    )


def _save_history(history, output):
    with (output / "history.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "accuracy"])
        writer.writeheader()
        writer.writerows(history)
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_scatter(
        x=[row["epoch"] for row in history],
        y=[row["loss"] for row in history],
        name="Training loss",
    )
    figure.add_scatter(
        x=[row["epoch"] for row in history],
        y=[row["accuracy"] for row in history],
        name="Training accuracy",
        yaxis="y2",
    )
    figure.update_layout(
        template="plotly_dark",
        title="Training history (not validation)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis2={
            "title": "Accuracy",
            "overlaying": "y",
            "side": "right",
            "range": [0, 1],
        },
    )
    figure.write_html(output / "training.html", include_plotlyjs=True)


def _write_predictions(path, ids, predictions, probabilities, classes, truth=None):
    fields = [
        "patient_id",
        *(["true_label"] if truth is not None else []),
        "predicted_label",
        *[f"probability_{label}" for label in classes],
    ]
    with Path(path).open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index, patient_id in enumerate(ids):
            writer.writerow(
                [
                    patient_id,
                    *([truth[index]] if truth is not None else []),
                    int(predictions[index]),
                    *map(float, probabilities[index]),
                ]
            )


def fit(
    matrix,
    output_dir,
    *,
    model_type,
    target_column="aki_label",
    features=None,
    train_fraction=0.8,
    seed=2026,
    epochs=1000,
    learning_rate=None,
    batch_size=32,
    tolerance=0.0,
    l2=0.0,
    hidden_layers=(16,),
    activation="relu",
    threshold=0.0,
    clamped=False,
):
    if model_type not in MODEL_TYPES:
        raise ValueError("Unknown model type")
    binary_rosenblatt = model_type == "rosenblatt_perceptron"
    if binary_rosenblatt and learning_rate is not None:
        raise ValueError(
            "Rosenblatt uses a fixed dimension-normalized learning rate; remove learning_rate"
        )
    if not binary_rosenblatt and clamped:
        raise ValueError("clamped is supported only by the Rosenblatt model")
    if binary_rosenblatt and target_column != "aki_label":
        raise ValueError("The Rosenblatt model is binary and requires aki_label")
    if not binary_rosenblatt and threshold != 0:
        raise ValueError("threshold is supported only by the Rosenblatt model")
    rows, features = read_matrix(matrix, target_column=target_column, features=features)
    training, validation, excluded = split_rows(
        rows, train_fraction, seed, target_column=target_column
    )
    transformer = Transformer.fit(training, features)
    x, xv = transformer.transform(training), transformer.transform(validation)
    y = [int(row[target_column]) for row in training]
    truth = [int(row[target_column]) for row in validation]
    if binary_rosenblatt:
        model, stats = train_rosenblatt(
            x,
            y,
            epochs=epochs,
            threshold=threshold,
            clamped=clamped,
        )
        history = stats.pop("history")
        artifact = {
            "schema_version": 1,
            "model_type": "binary_rosenblatt_perceptron",
            "labels": {"-1": "No AKI", "1": "AKI"},
            "weights": model.weights,
            "bias": model.bias,
            "threshold": model.threshold,
        }
        scores = [model.score(row) for row in xv]
        predictions = [1 if score >= 0 else -1 for score in scores]
    else:
        model = _probabilistic_model(
            model_type, len(x[0]), sorted(set(y)), seed, l2, hidden_layers, activation
        )
        rate = 0.05 if learning_rate is None else learning_rate
        model.fit(
            x,
            y,
            epochs=epochs,
            learning_rate=rate,
            batch_size=batch_size,
            tolerance=tolerance,
        )
        history = model.history_
        stats = {
            "epochs_run": len(history),
            "max_epochs": epochs,
            "learning_rate": rate,
            "batch_size": batch_size,
            "tolerance": tolerance,
            "seed": seed,
        }
        artifact = {
            "schema_version": 1,
            "model_type": model_type,
            "target_column": target_column,
            "model": model.to_dict(),
        }
        probabilities = model.predict_proba(xv)
        predictions = model.predict(xv)
    stats.update(
        {
            "train_fraction": train_fraction,
            "seed": seed,
            "training_patients": len(training),
            "validation_patients": len(validation),
            "excluded_unlabeled": excluded,
        }
    )
    artifact.update({"transformer": transformer.to_dict(), "training": stats})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids = [row["patient_id"] for row in validation]
    if binary_rosenblatt:
        save_training_plots(history, output / "plots/convergence")
        metrics = save_report(output / "validation", ids, truth, predictions, scores)
    else:
        _save_history(history, output)
        classes = model.classes_.tolist()
        scores = (
            probabilities[:, classes.index(1)].tolist()
            if target_column == "aki_label"
            else None
        )
        metrics = save_classification_report(
            output / "validation",
            truth,
            predictions,
            labels=TARGETS[target_column],
            scores=scores,
        )
        _write_predictions(
            output / "validation/predictions.csv",
            ids,
            predictions,
            probabilities,
            classes,
            truth,
        )
    (output / "model.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False), encoding="utf8"
    )
    (output / "split.json").write_text(
        json.dumps(
            {
                "training_patient_ids": [row["patient_id"] for row in training],
                "validation_patient_ids": ids,
                "excluded_unlabeled": excluded,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    return {
        "artifact": str(output / "model.json"),
        "metrics": metrics,
        "training": stats,
    }


def load_artifact(path):
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported model artifact schema")
    kind = payload.get("model_type")
    if kind == "logistic_regression":
        from src.models.logistic_regression import LogisticRegression

        model_class = LogisticRegression
    elif kind == "multilayer_perceptron":
        from src.models.neural_network import MultiLayerPerceptron

        model_class = MultiLayerPerceptron
    else:
        raise ValueError("Unsupported model type")
    try:
        target = payload["target_column"]
        if target not in TARGETS:
            raise ValueError("Unsupported model target")
        transformer = Transformer.from_dict(payload["transformer"])
        if any(is_forbidden_predictor(key) for key in transformer.features):
            raise ValueError("Artifact contains outcome predictors")
        model = model_class.from_dict(payload["model"])
        classes = model.classes_.tolist()
        if not set(classes).issubset(TARGETS[target]) or len(classes) < 2:
            raise ValueError("Model classes do not match target")
        model.predict_proba([transformer.transform_one({})])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed model artifact") from error
    return model, transformer, target


def predict_csv(artifact, input_csv, output_csv, *, expected_type=None):
    payload = json.loads(Path(artifact).read_text(encoding="utf8"))
    kind = payload.get("model_type") if isinstance(payload, dict) else None
    canonical = (
        "rosenblatt_perceptron" if kind == "binary_rosenblatt_perceptron" else kind
    )
    if expected_type is not None and canonical != expected_type:
        raise ValueError("Artifact model type does not match the prediction command")
    if canonical == "rosenblatt_perceptron":
        return predict_rosenblatt(artifact, input_csv, output_csv)
    model, transformer, target = load_artifact(artifact)
    rows, _ = read_matrix(
        input_csv,
        target_column=target,
        features=transformer.features,
        require_target=False,
    )
    if not rows:
        raise ValueError("Novel matrix contains no patients")
    x = transformer.transform(rows)
    probabilities, predictions = model.predict_proba(x), model.predict(x)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_predictions(
        output,
        [row["patient_id"] for row in rows],
        predictions,
        probabilities,
        model.classes_.tolist(),
    )
    return len(rows)
