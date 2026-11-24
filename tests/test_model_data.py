import csv

import pytest


def write_matrix(path, *, stages=False):
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "patient_id",
                "preoperative/x",
                "preoperative/group",
                "aki_label",
                "aki_stage",
                "label_status",
            ]
        )
        for i in range(24):
            stage = i // 6
            writer.writerow(
                [
                    f"synthetic_{i:02}",
                    i,
                    "A" if i % 2 else "B",
                    -1 if stage == 0 else 1,
                    stage,
                    "known",
                ]
            )
        writer.writerow(["unknown", 8, "A", "NA", "NA", "unknown"])


def test_patient_matrix_uses_explicit_target_and_never_outcome_features(tmp_path):
    from src.models.data import read_matrix, split_rows

    path = tmp_path / "matrix.csv"
    write_matrix(path)
    rows, features = read_matrix(path, target_column="aki_stage")
    assert features == ["preoperative/x", "preoperative/group"]
    training, validation, excluded = split_rows(
        rows, 0.75, 2026, target_column="aki_stage"
    )
    assert excluded == 1
    assert {int(row["aki_stage"]) for row in training} == {0, 1, 2, 3}
    assert {int(row["aki_stage"]) for row in validation} == {0, 1, 2, 3}
    assert len(training) + len(validation) + excluded == 25
    assert not {row["patient_id"] for row in training} & {
        row["patient_id"] for row in validation
    }
    assert split_rows(rows, 0.75, 2026, target_column="aki_stage") == (
        training,
        validation,
        excluded,
    )
    _, selected = read_matrix(path, features=["preoperative/group"])
    assert selected == ["preoperative/group"]


@pytest.mark.parametrize(
    "features",
    [
        ["aki_stage"],
        ["patient_id"],
        ["label_status"],
        ["postoperative/creatinine"],
        ["absent"],
        ["preoperative/x", "preoperative/x"],
    ],
)
def test_feature_selection_rejects_leakage_or_unknown_columns(tmp_path, features):
    from src.models.data import read_matrix

    path = tmp_path / "matrix.csv"
    write_matrix(path)
    with pytest.raises(ValueError):
        read_matrix(path, features=features)


def test_missing_targets_stay_unknown_and_invalid_targets_fail(tmp_path):
    from src.models.data import read_matrix, split_rows

    path = tmp_path / "matrix.csv"
    write_matrix(path)
    rows, _ = read_matrix(path)
    rows[-1]["aki_label"] = "2"
    with pytest.raises(ValueError, match="label"):
        split_rows(rows, 0.8, 1)
    rows[-1]["aki_label"] = ""
    assert split_rows(rows, 0.8, 1)[2] == 1


def test_shared_transformer_is_training_only_and_preserves_old_import():
    from src.models.data import Transformer
    from src.models.rosenblatt_perceptron.transform import (
        Transformer as LegacyTransformer,
    )

    assert LegacyTransformer is Transformer
    transformer = Transformer.fit(
        [{"x": "1", "group": "A"}, {"x": "3", "group": "B"}], ["x", "group"]
    )
    before = transformer.to_dict()
    assert transformer.transform_one({"x": "100", "group": "unseen"}) == [
        98.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert transformer.to_dict() == before
    assert Transformer.from_dict(before).transform_one({"x": "", "group": ""}) == [
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    ]


def test_matrix_rejects_duplicate_headers_or_patient_ids(tmp_path):
    from src.models.data import read_matrix

    path = tmp_path / "bad.csv"
    path.write_text("patient_id,x,x,aki_label\na,1,2,-1\n", encoding="utf8")
    with pytest.raises(ValueError, match="header"):
        read_matrix(path)
    path.write_text("patient_id,x,aki_label\na,1,-1\na,2,1\n", encoding="utf8")
    with pytest.raises(ValueError, match="IDs"):
        read_matrix(path)
