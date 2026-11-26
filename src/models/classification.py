"""Patient-level orchestration shared by classification baselines."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

from src.models.contracts import TARGETS, is_forbidden_predictor
from src.models.data import Transformer, read_matrix, split_rows
from src.models.evaluation.report import save_classification_report
from src.models.factory import restore_model
from src.models.rosenblatt_perceptron.core import train as train_rosenblatt
from src.models.rosenblatt_perceptron.execution.predict import (
    predict_csv as predict_rosenblatt,
)
from src.models.rosenblatt_perceptron.plots.charts import save_training_plots
from src.models.rosenblatt_perceptron.validation.report import save_report

MODEL_TYPES = (
    "rosenblatt_perceptron",
    "logistic_regression",
    "multilayer_perceptron",
    "multi_output_perceptron",
)


def _model_constructor(
    kind,
    width,
    classes,
    seed,
    l2,
    hidden_layers,
    activation,
    hidden_activations,
    init_methods,
):
    if kind == "logistic_regression":
        return {"l2": l2, "seed": seed}
    if l2 != 0:
        raise ValueError("l2 is currently supported only by logistic regression")
    if kind == "multi_output_perceptron":
        return {"seed": seed}
    if hidden_activations is not None:
        if activation is not None:
            raise ValueError("Use activation or hidden_activations, not both")
        activations = list(hidden_activations)
        if len(activations) != len(hidden_layers):
            raise ValueError(
                "hidden_activations must have one activation per hidden layer"
            )
    else:
        activations = ["relu" if activation is None else activation] * len(
            hidden_layers
        )
    return {
        "shape": [width, *hidden_layers, len(classes)],
        "activation_functions": [*activations, "softmax"],
        "init_methods": init_methods,
        "seed": seed,
    }


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


def _write_predictions(
    path, ids, predictions, probabilities, classes, truth=None, *, probabilistic=True
):
    fields = [
        "patient_id",
        *(["true_label"] if truth is not None else []),
        "predicted_label",
        *[
            f"{'probability' if probabilistic else 'score'}_{label}"
            for label in classes
        ],
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


def _configured_training(model_type, model_config, profile, overrides):
    values = {
        "epochs": 1000,
        "learning_rate": 0.05,
        "batch_size": 32,
        "tolerance": 0.0,
        "l2": 0.0,
        "hidden_layers": (16,),
        "activation": None,
        "hidden_activations": None,
        "init_methods": None,
    }
    if model_config is None:
        if profile is not None:
            raise ValueError("profile requires model_config")
        if model_type == "multi_output_perceptron" and any(
            value is not None
            for key, value in overrides.items()
            if key not in {"epochs", "learning_rate"}
        ):
            raise ValueError(
                "multi_output_perceptron supports only online epochs and learning_rate options"
            )
        values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        return values, {}
    if any(value is not None for value in overrides.values()):
        raise ValueError(
            "Model INI is authoritative; cannot combine it with architecture or training overrides"
        )
    from src.models.config import (
        LogisticSettings,
        MLPSettings,
        load_classifier_settings,
    )

    settings = load_classifier_settings(
        model_config, model_type=model_type, profile=profile
    )
    model = settings.model
    values.update(asdict(model.training))
    if isinstance(model, MLPSettings):
        values.update(
            hidden_layers=model.hidden_layers,
            hidden_activations=model.hidden_activations,
            init_methods=model.init_methods,
        )
    elif isinstance(model, LogisticSettings):
        values["l2"] = model.l2
    return values, asdict(settings.ray)


def fit(
    matrix,
    output_dir,
    *,
    model_type,
    target_column="aki_label",
    features=None,
    train_fraction=0.8,
    seed=2026,
    epochs=None,
    learning_rate=None,
    batch_size=None,
    tolerance=None,
    l2=None,
    hidden_layers=None,
    activation=None,
    hidden_activations=None,
    init_methods=None,
    model_config=None,
    profile=None,
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
    if binary_rosenblatt and (model_config is not None or profile is not None):
        raise ValueError(
            "Model INI profiles are not used by the course Rosenblatt model"
        )
    values, ray_options = _configured_training(
        model_type,
        model_config,
        profile,
        {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "tolerance": tolerance,
            "l2": l2,
            "hidden_layers": hidden_layers,
            "activation": activation,
            "hidden_activations": hidden_activations,
            "init_methods": init_methods,
        },
    )
    epochs, learning_rate = values["epochs"], values["learning_rate"]
    batch_size, tolerance = values["batch_size"], values["tolerance"]
    l2, hidden_layers = values["l2"], values["hidden_layers"]
    activation, hidden_activations = values["activation"], values["hidden_activations"]
    init_methods = values["init_methods"]
    probabilistic = model_type != "multi_output_perceptron"
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
        from src.models.ray_backend import fit_with_ray

        constructor = _model_constructor(
            model_type,
            len(x[0]),
            sorted(set(y)),
            seed,
            l2,
            hidden_layers,
            activation,
            hidden_activations,
            init_methods,
        )
        rate = 0.05 if learning_rate is None else learning_rate
        fit_options = {"epochs": epochs, "learning_rate": rate}
        if model_type != "multi_output_perceptron":
            fit_options.update(batch_size=batch_size, tolerance=tolerance)
        payload, execution = fit_with_ray(
            model_type, x, y, constructor, fit_options, **ray_options
        )
        model = restore_model(model_type, payload)
        history = model.history_
        stats = {
            "epochs_run": len(history),
            "max_epochs": epochs,
            "learning_rate": rate,
            "seed": seed,
            "execution": execution,
            "loss": "cross_entropy" if probabilistic else "mistake_rate",
        }
        if probabilistic:
            stats.update(batch_size=batch_size, tolerance=tolerance)
        else:
            stats.update(updates=model.updates_, converged=model.converged_)
        artifact = {
            "schema_version": 1,
            "model_type": model_type,
            "target_column": target_column,
            "model": model.to_dict(),
        }
        probabilities = (
            model.predict_proba(xv) if probabilistic else model.decision_function(xv)
        )
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
            (
                probabilities[:, classes.index(1)]
                if probabilistic
                else probabilities[:, classes.index(1)]
                - probabilities[:, classes.index(-1)]
            ).tolist()
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
            probabilistic=probabilistic,
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
    if kind not in MODEL_TYPES or kind == "rosenblatt_perceptron":
        raise ValueError("Unsupported model type")
    try:
        target = payload["target_column"]
        if target not in TARGETS:
            raise ValueError("Unsupported model target")
        transformer = Transformer.from_dict(payload["transformer"])
        if any(is_forbidden_predictor(key) for key in transformer.features):
            raise ValueError("Artifact contains outcome predictors")
        model = restore_model(kind, payload["model"])
        classes = model.classes_.tolist()
        if not set(classes).issubset(TARGETS[target]) or len(classes) < 2:
            raise ValueError("Model classes do not match target")
        if kind == "multi_output_perceptron":
            model.decision_function([transformer.transform_one({})])
        else:
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
    probabilistic = canonical != "multi_output_perceptron"
    probabilities = (
        model.predict_proba(x) if probabilistic else model.decision_function(x)
    )
    predictions = model.predict(x)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_predictions(
        output,
        [row["patient_id"] for row in rows],
        predictions,
        probabilities,
        model.classes_.tolist(),
        probabilistic=probabilistic,
    )
    return len(rows)
