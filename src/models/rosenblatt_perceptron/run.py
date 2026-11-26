"""CLI train/evaluate and novel-data execution."""

import json

from src.models.rosenblatt_perceptron.execution.predict import predict_csv
from src.models.rosenblatt_perceptron.training import fit


def train_command(args):
    result = fit(
        args.input,
        args.output,
        train_fraction=args.train_fraction,
        seed=args.seed,
        epochs=args.max_epochs,
        threshold=args.threshold,
        clamped=args.clamped,
    )
    print(json.dumps(result, indent=2))
    return 0


def predict_command(args):
    count = predict_csv(args.model, args.input, args.output)
    print(f"Wrote {count} predictions to {args.output}")
    return 0


def register_parser(subparsers):
    t = subparsers.add_parser(
        "rosenblatt-train", help="Train binary AKI perceptron and evaluate holdout"
    )
    t.add_argument("--input", default="results/inputsModel/patients_labeled.csv")
    t.add_argument("--output", default="results/models/rosenblatt_perceptron")
    t.add_argument("--train-fraction", type=float, default=0.8)
    t.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed for the patient split only; epochs keep fixed example order",
    )
    t.add_argument("--max-epochs", type=int, default=100)
    t.add_argument(
        "--clamped",
        action="store_true",
        help="Append a constant +1 input; effective learning rate is 1/(N+1)",
    )
    t.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Finite signed-margin update threshold; convergence always requires margins > 0",
    )
    t.set_defaults(handler=train_command)
    x = subparsers.add_parser(
        "rosenblatt-predict", help="Predict novel patients using frozen model"
    )
    x.add_argument("--model", default="results/models/rosenblatt_perceptron/model.json")
    x.add_argument("--input", required=True)
    x.add_argument(
        "--output", default="results/models/rosenblatt_perceptron/novel_predictions.csv"
    )
    x.set_defaults(handler=predict_command)
