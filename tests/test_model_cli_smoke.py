"""Isolated CLI smokes: no active clinical configurations or cohort files."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", *map(str, args)],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf8",
        env={**os.environ, "PYTHONIOENCODING": "utf8"},
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize(
    "prefix,target", [("logistic", "aki_stage"), ("mlp", "aki_label")]
)
def test_real_cli_training_and_prediction(tmp_path, prefix, target):
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "preoperative/x", "aki_label", "aki_stage"])
        for i in range(24):
            writer.writerow([f"synthetic_{i}", i, -1 if i < 12 else 1, i // 6])
    output = tmp_path / prefix
    _run(
        f"{prefix}-train",
        "--input",
        matrix,
        "--output",
        output,
        "--target-column",
        target,
        "--features",
        "preoperative/x",
        "--max-epochs",
        12,
    )
    artifact = json.loads((output / "model.json").read_text())
    assert artifact["target_column"] == target
    assert (
        artifact["training"]["training_patients"]
        + artifact["training"]["validation_patients"]
        == 24
    )
    novel = tmp_path / "novel.csv"
    novel.write_text("patient_id,preoperative/x\nnew_synthetic,4\n", encoding="utf8")
    destination = output / "novel.csv"
    _run(
        f"{prefix}-predict",
        "--model",
        output / "model.json",
        "--input",
        novel,
        "--output",
        destination,
    )
    with destination.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["patient_id"] == "new_synthetic"
    assert "true_label" not in rows[0]
    assert sum(
        float(value) for key, value in rows[0].items() if key.startswith("probability_")
    ) == pytest.approx(1)
