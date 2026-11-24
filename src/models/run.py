"""CLI commands for probabilistic classification baselines."""

import json

from src.models.classification import fit, predict_csv
from src.models.contracts import HIDDEN_ACTIVATIONS


def train_command(args):
    options = (
        {"l2": args.l2}
        if args.model_type == "logistic_regression"
        else {
            "hidden_layers": args.hidden_layers,
            "activation": args.activation,
        }
    )
    result = fit(
        args.input,
        args.output,
        model_type=args.model_type,
        target_column=args.target_column,
        features=args.features,
        train_fraction=args.train_fraction,
        seed=args.seed,
        epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        tolerance=args.tolerance,
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
    ]:
        train = commands.add_parser(
            f"{prefix}-train",
            help=f"Train/evaluate {kind} on a patient-disjoint holdout",
        )
        train.add_argument(
            "--input", default="results/inputsModel/patients_labeled.csv"
        )
        train.add_argument("--output", default=f"results/models/{kind}")
        train.add_argument(
            "--target-column", choices=["aki_label", "aki_stage"], default="aki_label"
        )
        train.add_argument(
            "--features", nargs="+", help="Explicit feature columns, in model order"
        )
        train.add_argument("--train-fraction", type=float, default=0.8)
        train.add_argument("--seed", type=int, default=2026)
        train.add_argument("--max-epochs", type=int, default=1000)
        train.add_argument("--learning-rate", type=float, default=0.05)
        train.add_argument("--batch-size", type=int, default=32)
        train.add_argument("--tolerance", type=float, default=0.0)
        if prefix == "logistic":
            train.add_argument(
                "--l2",
                type=float,
                default=0.0,
                help="Weight penalty; intercept is not regularized",
            )
        else:
            train.add_argument("--hidden-layers", type=int, nargs="+", default=[16])
            train.add_argument(
                "--activation",
                choices=HIDDEN_ACTIVATIONS,
                default="relu",
            )
        train.set_defaults(handler=train_command, model_type=kind)
        predict = commands.add_parser(
            f"{prefix}-predict", help=f"Predict novel patients with a frozen {kind}"
        )
        predict.add_argument("--model", required=True)
        predict.add_argument("--input", required=True)
        predict.add_argument("--output", required=True)
        predict.set_defaults(handler=predict_command, model_type=kind)
