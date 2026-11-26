"""CLI for Ray-backed classifiers and runtime-independent prediction."""

import json
from pathlib import Path

from src.models.classification import fit, predict_csv
from src.models.contracts import HIDDEN_ACTIVATIONS


def train_command(args):
    options = {
        name: getattr(args, name, None)
        for name in (
            "learning_rate",
            "batch_size",
            "tolerance",
            "l2",
            "hidden_layers",
            "activation",
            "hidden_activations",
            "init_methods",
        )
    }
    result = fit(
        args.input,
        args.output,
        model_type=args.model_type,
        target_column=args.target_column,
        features=args.features,
        train_fraction=args.train_fraction,
        seed=args.seed,
        epochs=args.max_epochs,
        model_config=args.config,
        profile=args.profile,
        **options,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def predict_command(args):
    count = predict_csv(
        args.model, args.input, args.output, expected_type=args.model_type
    )
    print(f"Wrote {count} predictions to {args.output}")
    return 0


def register_parser(commands):
    for prefix, kind in [
        ("logistic", "logistic_regression"),
        ("mlp", "multilayer_perceptron"),
        ("multioutput", "multi_output_perceptron"),
    ]:
        train = commands.add_parser(
            f"{prefix}-train",
            help=f"Train/evaluate {kind} in a real local Ray task",
        )
        train.add_argument(
            "--input", default="results/inputsModel/patients_labeled.csv"
        )
        train.add_argument("--output", default=f"results/models/{kind}")
        train.add_argument(
            "--config",
            type=Path,
            help="Model INI; authoritative for architecture, training and Ray resources",
        )
        train.add_argument(
            "--profile", help="INI section; defaults to the selected model type"
        )
        train.add_argument(
            "--target-column", choices=["aki_label", "aki_stage"], default="aki_label"
        )
        train.add_argument(
            "--features", nargs="+", help="Explicit feature columns, in model order"
        )
        train.add_argument("--train-fraction", type=float, default=0.8)
        train.add_argument("--seed", type=int, default=2026)
        train.add_argument(
            "--max-epochs", type=int, help="Without INI only; default 1000"
        )
        train.add_argument(
            "--learning-rate", type=float, help="Without INI only; default 0.05"
        )
        if prefix != "multioutput":
            train.add_argument(
                "--batch-size", type=int, help="Without INI only; default 32"
            )
            train.add_argument(
                "--tolerance", type=float, help="Without INI only; default 0"
            )
        if prefix == "logistic":
            train.add_argument(
                "--l2",
                type=float,
                help="Without INI only; weight penalty, not intercept; default 0",
            )
        elif prefix == "mlp":
            train.add_argument(
                "--hidden-layers",
                type=int,
                nargs="*",
                help="Without INI only; hidden widths, default 16; empty gives no hidden layers",
            )
            train.add_argument(
                "--activation",
                choices=HIDDEN_ACTIVATIONS,
                help="Without INI only; broadcast hidden activation, default relu",
            )
            train.add_argument(
                "--hidden-activations",
                choices=HIDDEN_ACTIVATIONS,
                nargs="*",
                help="Without INI only; one per hidden layer; cannot combine with --activation",
            )
            train.add_argument(
                "--init-methods",
                choices=["he", "xavier", "zero"],
                nargs="+",
                help="Without INI only; one initializer per hidden AND output layer",
            )
        train.set_defaults(handler=train_command, model_type=kind)
        predict = commands.add_parser(
            f"{prefix}-predict", help=f"Predict with a frozen {kind}; no Ray startup"
        )
        predict.add_argument("--model", required=True)
        predict.add_argument("--input", required=True)
        predict.add_argument("--output", required=True)
        predict.set_defaults(handler=predict_command, model_type=kind)
