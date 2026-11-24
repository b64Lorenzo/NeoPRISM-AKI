import pytest

from src.cli import build_parser


@pytest.mark.parametrize(
    "command,kind",
    [("logistic-train", "logistic_regression"), ("mlp-train", "multilayer_perceptron")],
)
def test_new_model_training_commands_share_explicit_target_and_features(command, kind):
    args = build_parser().parse_args(
        [
            command,
            "--input",
            "matrix.csv",
            "--output",
            "trained",
            "--target-column",
            "aki_stage",
            "--features",
            "preoperative/age",
            "--seed",
            "7",
            "--max-epochs",
            "20",
            "--batch-size",
            "5",
        ]
    )
    assert args.model_type == kind
    assert args.target_column == "aki_stage"
    assert args.features == ["preoperative/age"]
    assert args.max_epochs == 20
    assert args.batch_size == 5
    assert args.seed == 7


@pytest.mark.parametrize("command", ["logistic-predict", "mlp-predict"])
def test_new_prediction_commands_require_explicit_paths(command):
    args = build_parser().parse_args(
        [
            command,
            "--model",
            "model.json",
            "--input",
            "novel.csv",
            "--output",
            "predicted.csv",
        ]
    )
    assert args.model == "model.json"
    assert args.input == "novel.csv"
    assert args.output == "predicted.csv"
