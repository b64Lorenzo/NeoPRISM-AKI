import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.cli import build_parser
from src.models.classification import fit, load_artifact, predict_csv


def profile(path, kind, *, extra=""):
    path.write_text(
        f"""[ray]
num_cpus = 1
cpus_per_task = 1
timeout_seconds = 60
[experiment]
model_type = {kind}
epochs = 5
learning_rate = 0.05
{extra}
""",
        encoding="utf8",
    )
    return path


def matrix(path):
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "preoperative/x", "aki_stage"])
        writer.writerows([f"synthetic_{i}", i, i // 4] for i in range(16))
    return path


@pytest.mark.parametrize(
    "kind", ["multilayer_perceptron", "multi_output_perceptron", "logistic_regression"]
)
def test_model_profile_executes_real_ray_and_artifact_prediction_needs_no_ray(
    tmp_path, kind, monkeypatch
):
    import ray

    extra = (
        "hidden_layers = 8, 4\nhidden_activations = relu, tanh\ninit_methods = he, xavier, xavier"
        if kind == "multilayer_perceptron"
        else ""
    )
    ini = profile(tmp_path / "models.ini", kind, extra=extra)
    output = tmp_path / "output"
    result = fit(
        matrix(tmp_path / "matrix.csv"),
        output,
        model_type=kind,
        target_column="aki_stage",
        model_config=ini,
        profile="experiment",
    )
    assert result["training"]["execution"]["backend"] == "ray"
    assert result["training"]["execution"]["worker_pid"] != os.getpid()
    assert not ray.is_initialized()
    artifact = output / "model.json"
    saved = json.loads(artifact.read_text())
    if kind == "multilayer_perceptron":
        assert saved["model"]["shape"] == [2, 8, 4, 4]
        assert saved["model"]["activation_functions"] == ["relu", "tanh", "softmax"]
    model, transformer, target = load_artifact(artifact)
    assert model.classes_.tolist() == [0, 1, 2, 3]
    assert list(transformer.features) == ["preoperative/x"]
    assert target == "aki_stage"
    novel = tmp_path / "novel.csv"
    novel.write_text("patient_id,preoperative/x\nnew_a,2\nnew_b,12\n")

    def forbidden(**kwargs):
        raise AssertionError("artifact inference must not start Ray")

    monkeypatch.setattr(ray, "init", forbidden)
    predictions = tmp_path / "predictions.csv"
    assert predict_csv(artifact, novel, predictions, expected_type=kind) == 2
    with predictions.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    prefix = "score_" if kind == "multi_output_perceptron" else "probability_"
    assert [name for name in rows[0] if name.startswith(prefix)] == [
        f"{prefix}{i}" for i in range(4)
    ]
    assert (
        not any(name.startswith("probability_") for name in rows[0])
        if kind == "multi_output_perceptron"
        else True
    )
    values = np.array([[float(row[f"{prefix}{i}"]) for i in range(4)] for row in rows])
    assert np.isfinite(values).all()
    assert [int(row["predicted_label"]) for row in rows] == np.argmax(
        values, axis=1
    ).tolist()
    if kind != "multi_output_perceptron":
        np.testing.assert_allclose(values.sum(axis=1), 1)
        assert (values >= 0).all()
    import subprocess
    import sys

    probe = """import sys, importlib.abc
class NoRay(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ray' or fullname.startswith('ray.'):
            raise ImportError('Ray is unavailable during artifact-only prediction')
sys.meta_path.insert(0, NoRay())
from src.models.classification import predict_csv
assert predict_csv(sys.argv[1], sys.argv[2], sys.argv[3]) == 2
assert 'ray' not in sys.modules
print('prediction without Ray passed')
"""
    detached = tmp_path / "detached.csv"
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(artifact), str(novel), str(detached)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert detached.read_bytes() == predictions.read_bytes()


@pytest.mark.parametrize(
    "prefix, kind",
    [
        ("mlp", "multilayer_perceptron"),
        ("multioutput", "multi_output_perceptron"),
        ("logistic", "logistic_regression"),
    ],
)
def test_training_cli_accepts_model_profile_without_implicit_overrides(prefix, kind):
    args = build_parser().parse_args(
        [prefix + "-train", "--config", "models.ini", "--profile", "experiment"]
    )
    assert args.model_type == kind
    assert args.config == Path("models.ini")
    assert args.profile == "experiment"
    assert args.max_epochs is None
    assert args.learning_rate is None


def test_explicit_default_cli_override_is_still_a_conflict_with_ini(tmp_path):
    ini = profile(tmp_path / "models.ini", "multilayer_perceptron")
    args = build_parser().parse_args(
        [
            "mlp-train",
            "--config",
            str(ini),
            "--profile",
            "experiment",
            "--max-epochs",
            "1000",
            "--input",
            str(tmp_path / "missing.csv"),
        ]
    )
    with pytest.raises(ValueError, match="authoritative|cannot.*combin"):
        args.handler(args)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "options", [{"hidden_layers": [4]}, {"batch_size": 32}, {"tolerance": 0.0}]
)
def test_hard_perceptron_rejects_unrelated_optimizer_or_architecture_options(
    tmp_path, options
):
    with pytest.raises(ValueError, match="multi.output|online|unsupported"):
        fit(
            tmp_path / "absent.csv",
            tmp_path / "output",
            model_type="multi_output_perceptron",
            **options,
        )
    assert not (tmp_path / "output").exists()


def test_flag_only_multioutput_training_also_uses_ray_and_has_no_minibatch_metadata(
    tmp_path,
):
    result = fit(
        matrix(tmp_path / "matrix.csv"),
        tmp_path / "output",
        model_type="multi_output_perceptron",
        target_column="aki_stage",
        epochs=2,
    )
    assert result["training"]["execution"]["backend"] == "ray"
    assert result["training"]["execution"]["worker_pid"] != os.getpid()
    assert "batch_size" not in result["training"]
    assert "tolerance" not in result["training"]
    assert result["training"]["loss"] == "mistake_rate"
